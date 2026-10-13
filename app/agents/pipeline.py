"""Orchestrator: detect language -> rewrite -> route -> retrieve -> select -> generate -> verify."""

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.agents import rewriter, router, verifier
from app.core.config import Settings
from app.core.lang import detect_language, language_name, refusal_message
from app.core.schemas import Passage
from app.llm.client import LLMClient, LLMError
from app.memory.session import SessionMemory
from app.okf.graph import OKFIndex
from app.rag.store import VectorStore

log = logging.getLogger(__name__)

GROUNDED_SYSTEM = """You are a careful multilingual assistant. Answer using ONLY the numbered context passages provided.
Rules:
1. Cite the passages that support each statement inline, like [1] or [2][3].
2. If the passages do not contain the answer, say you don't have enough information. Never guess or use outside knowledge.
3. Passages may be in a different language from the question; translate faithfully.
4. Write the entire reply in {lang}.
5. Be concise and precise."""

SMALLTALK_SYSTEM = ("You are a friendly multilingual assistant. Reply in {lang}. Keep it brief. "
                    "If asked for facts you are unsure about, say so instead of guessing.")


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


@dataclass
class Prepared:
    session_id: str
    message: str
    language: str
    query: str
    route: str
    route_reason: str
    passages: list[Passage]
    raw: dict
    history: list[dict]
    timings: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, settings: Settings, llm: LLMClient, store: VectorStore,
                 okf: OKFIndex, memory: SessionMemory):
        self.s, self.llm, self.store, self.okf, self.memory = settings, llm, store, okf, memory

    # ---------- retrieval ----------
    async def retrieve(self, query: str, route: str) -> dict:
        raw: dict[str, list[Passage]] = {"vector": [], "okf": []}
        if route in ("vector", "both"):
            raw["vector"] = await asyncio.to_thread(self.store.search, query, self.s.top_k * 2)
        if route in ("okf", "both"):
            raw["okf"] = await asyncio.to_thread(self.okf.search, query, 5)
        return raw

    def select(self, raw: dict) -> list[Passage]:
        vec = [p for p in raw["vector"] if p.score >= self.s.min_score][: self.s.top_k]
        okf = [p for p in raw["okf"] if p.score >= self.s.okf_min_score][:3]
        okf = self.okf.expand(okf, limit=self.s.okf_neighbors)
        return okf + vec

    async def prepare(self, message: str, session_id: str | None = None,
                      language: str | None = None) -> Prepared:
        timings: dict[str, float] = {}
        session_id = session_id or uuid.uuid4().hex
        lang = language or detect_language(message)
        history = self.memory.history(session_id, self.s.history_turns)

        t = time.perf_counter()
        query = await rewriter.rewrite(self.llm, message, history)
        timings["rewrite"] = _ms(t)

        t = time.perf_counter()
        decision = await router.route(self.llm, query, self.okf)
        timings["route"] = _ms(t)

        t = time.perf_counter()
        raw = await self.retrieve(query, decision.route)
        passages = self.select(raw)
        timings["retrieve"] = _ms(t)

        return Prepared(session_id, message, lang, query, decision.route, decision.reason,
                        passages, raw, history, timings)

    # ---------- generation ----------
    def build_messages(self, prep: Prepared) -> list[dict]:
        lang = language_name(prep.language)
        if prep.route == "none":
            system, user = SMALLTALK_SYSTEM.format(lang=lang), prep.message
        else:
            ctx = "\n\n".join(
                f"[{i}] (source: {p.source}{', ' + p.meta['section'] if p.meta.get('section') else ''})\n{p.text}"
                for i, p in enumerate(prep.passages, 1))
            user = f"Context passages:\n{ctx}\n\nQuestion: {prep.message}"
            if prep.query != prep.message:
                user += f"\n(Standalone form: {prep.query})"
            system = GROUNDED_SYSTEM.format(lang=lang)
        return [{"role": "system", "content": system}, *prep.history, {"role": "user", "content": user}]

    def _no_llm_answer(self, prep: Prepared, note: str) -> str:
        if prep.route == "none":
            return f"({note})"
        lines = [f"({note} Showing the most relevant retrieved passages instead.)"]
        for i, p in enumerate(prep.passages, 1):
            lines.append(f"\n[{i}] {p.source}: {p.text[:350].strip()}")
        return "\n".join(lines)

    def _shortcut(self, prep: Prepared) -> str | None:
        if prep.route != "none" and not prep.passages:
            return refusal_message(prep.language)
        if not self.llm.available:
            return self._no_llm_answer(prep, "LLM not configured.")
        return None

    async def generate(self, prep: Prepared) -> str:
        t = time.perf_counter()
        try:
            answer = self._shortcut(prep)
            if answer is None:
                try:
                    answer = await self.llm.chat(self.build_messages(prep))
                except LLMError as e:
                    log.warning("LLM failure: %s", e)
                    answer = self._no_llm_answer(prep, "The language model is unavailable.")
            return re.sub(r"【(\d+)】", r"[\1]", answer)
        finally:
            prep.timings["generate"] = _ms(t)

    async def generate_stream(self, prep: Prepared) -> AsyncIterator[str]:
        t = time.perf_counter()
        short = self._shortcut(prep)
        if short is not None:
            yield short
        else:
            try:
                async for tok in self.llm.stream(self.build_messages(prep)):
                    yield tok
            except LLMError as e:
                log.warning("LLM stream failure: %s", e)
                yield "\n" + self._no_llm_answer(prep, "The language model is unavailable.")
        prep.timings["generate"] = _ms(t)

    async def verify(self, prep: Prepared, answer: str) -> dict | None:
        if not (self.s.enable_verifier and prep.passages and prep.route != "none" and self.llm.available):
            return None
        t = time.perf_counter()
        result = await verifier.verify(self.llm, answer, prep.passages)
        prep.timings["verify"] = _ms(t)
        return result

    def remember(self, prep: Prepared, answer: str) -> None:
        self.memory.add(prep.session_id, "user", prep.message)
        self.memory.add(prep.session_id, "assistant", answer)

    def meta(self, prep: Prepared) -> dict:
        return {
            "session_id": prep.session_id, "language": prep.language, "route": prep.route,
            "route_reason": prep.route_reason, "rewritten_query": prep.query,
            "grounded": bool(prep.passages), "sources": [p.public() for p in prep.passages],
        }

    async def chat(self, message: str, session_id: str | None = None,
                   language: str | None = None) -> dict:
        prep = await self.prepare(message, session_id, language)
        answer = await self.generate(prep)
        verification = await self.verify(prep, answer)
        self.remember(prep, answer)
        return {**self.meta(prep), "answer": answer, "verification": verification,
                "timings_ms": prep.timings}
