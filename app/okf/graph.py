"""OKF-inspired knowledge layer.
Convention (our own, deliberately simple): one concept per Markdown file with YAML front-matter
    ---
    id: rag                      # unique, stable
    title: Retrieval-Augmented Generation
    type: concept                # concept | process | entity | faq | ...
    lang: en
    aliases: [RAG, रैग]          # synonyms / translations -> multilingual lookup
    tags: [llm, retrieval]
    related: [vector-database]   # edges of the knowledge graph
    sources: [https://...]
    ---
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np
import yaml

from app.core.schemas import Passage
from app.rag.embedder import Embedder

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
_STOP = {"the", "and", "for", "what", "how", "why", "are", "is", "of", "to", "in", "a", "an", "does", "do"}


@dataclass
class OKFNode:
    id: str
    title: str
    body: str
    type: str = "concept"
    lang: str = "en"
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def labels(self) -> list[str]:
        return [self.title, self.id, *self.aliases]

    def search_text(self) -> str:
        return f"{self.title}. {' '.join(self.aliases)}. {' '.join(self.tags)}. {self.body[:400]}"


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def parse_okf_file(path: str) -> OKFNode:
    raw = open(path, encoding="utf-8").read()
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _FM.match(raw)
    meta, body = ({}, raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    return OKFNode(
        id=str(meta.get("id", stem)),
        title=str(meta.get("title", stem)),
        body=body.strip(),
        type=str(meta.get("type", "concept")),
        lang=str(meta.get("lang", "en")),
        aliases=_as_list(meta.get("aliases")),
        tags=_as_list(meta.get("tags")),
        related=_as_list(meta.get("related")),
        sources=_as_list(meta.get("sources")),
        path=path,
    )


def _has_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I) is not None


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if (len(t) > 2 or not t.isascii()) and t not in _STOP}


class OKFIndex:
    def __init__(self, directory: str, embedder: Embedder):
        self.dir = directory
        self.embedder = embedder
        self.nodes: dict[str, OKFNode] = {}
        self.backlinks: dict[str, set[str]] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self.load()

    def load(self) -> int:
        self.nodes, self.backlinks, self._matrix = {}, {}, None
        if os.path.isdir(self.dir):
            for root, _, files in os.walk(self.dir):
                for f in sorted(files):
                    if f.endswith(".md"):
                        node = parse_okf_file(os.path.join(root, f))
                        self.nodes[node.id] = node
        for n in self.nodes.values():
            for r in n.related:
                self.backlinks.setdefault(r, set()).add(n.id)
        return len(self.nodes)

    def catalog(self, limit: int = 60) -> list[str]:
        return [n.title for n in list(self.nodes.values())[:limit]]

    def neighbors(self, node_id: str) -> list[str]:
        n = self.nodes.get(node_id)
        if not n:
            return []
        out = [r for r in n.related if r in self.nodes]
        out += [b for b in sorted(self.backlinks.get(node_id, ())) if b in self.nodes and b not in out]
        return out

    def exact_match(self, query: str) -> OKFNode | None:
        for n in self.nodes.values():
            if any(_has_phrase(query, label) for label in n.labels):
                return n
        return None

    def _ensure_matrix(self):
        if self._matrix is None and self.nodes:
            self._ids = list(self.nodes)
            vecs = self.embedder.embed_documents([self.nodes[i].search_text() for i in self._ids])
            self._matrix = np.array(vecs, dtype=np.float32)

    def _lexical(self, node: OKFNode, query: str, q_tokens: set[str]) -> float:
        raw = 0.0
        if any(_has_phrase(query, label) for label in node.labels):
            raw += 3.0
        tag_hits = sum(1 for t in node.tags if _has_phrase(query, t))
        raw += min(3.0, 1.5 * tag_hits)
        if q_tokens:
            raw += 2.0 * len(q_tokens & _tokens(node.title + " " + node.body)) / len(q_tokens)
        return min(1.0, raw / 4.0)

    def search(self, query: str, k: int = 5) -> list[Passage]:
        """Hybrid lexical + semantic scoring over concept nodes. Returns raw (unfiltered) top-k."""
        if not self.nodes:
            return []
        self._ensure_matrix()
        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        sims = self._matrix @ qv
        q_tokens = _tokens(query)
        scored = []
        for i, nid in enumerate(self._ids):
            node = self.nodes[nid]
            lex = self._lexical(node, query, q_tokens)
            sem = max(0.0, float(sims[i]))
            scored.append((0.6 * lex + 0.4 * sem, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._to_passage(n, s) for s, n in scored[:k]]

    def expand(self, passages: list[Passage], limit: int = 2) -> list[Passage]:
        """Add 1-hop neighbours of the matched nodes (graph traversal)."""
        seen = {p.meta.get("node_id") for p in passages}
        out = list(passages)
        added = 0
        for p in passages:
            for nid in self.neighbors(p.meta.get("node_id", "")):
                if nid in seen or added >= limit:
                    continue
                seen.add(nid)
                out.append(self._to_passage(self.nodes[nid], p.score * 0.6, via=p.meta["node_id"]))
                added += 1
        return out

    @staticmethod
    def _to_passage(node: OKFNode, score: float, via: str | None = None) -> Passage:
        meta = {"node_id": node.id, "type": node.type, "lang": node.lang, "related": node.related,
                "sources": node.sources, "section": node.title}
        if via:
            meta["via"] = via
        return Passage(id=f"okf:{node.id}", text=f"{node.title}\n{node.body}"[:1800],
                       source=f"okf:{node.id}", kind="okf", score=score, meta=meta)
