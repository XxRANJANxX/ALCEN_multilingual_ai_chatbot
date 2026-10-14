import re
from dataclasses import dataclass

from app.core.jsonutil import parse_json
from app.llm.client import LLMClient, LLMError
from app.okf.graph import OKFIndex

VALID = {"okf", "vector", "both", "none"}
CHITCHAT = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|bye|namaste|नमस्ते|धन्यवाद|hola|gracias|"
    r"bonjour|merci|hallo|danke)\W*$", re.I)

PROMPT = """You route a user question to a knowledge source. Reply with JSON only:
{{"route": "okf" | "vector" | "both" | "none", "reason": "<short>"}}
- okf: definitions / concepts that match the curated concept list below
- vector: detailed content likely in uploaded documents (guides, reports, manuals)
- both: unsure, or the question could use either
- none: ONLY greetings, thanks, or pure language tasks (translate/rewrite this text). Any factual question must NOT be "none".
Curated concepts: {concepts}"""


@dataclass
class RouteDecision:
    route: str
    reason: str


async def route(llm: LLMClient, query: str, okf: OKFIndex) -> RouteDecision:
    if CHITCHAT.match(query):
        return RouteDecision("none", "greeting / small talk")
    exact = okf.exact_match(query)
    decision = None
    if llm.available:
        try:
            data = parse_json(await llm.chat(
                [{"role": "system", "content": PROMPT.format(concepts=", ".join(okf.catalog()) or "(none)")},
                 {"role": "user", "content": query}],
                temperature=0, max_tokens=120))
            if data.get("route") in VALID:
                decision = RouteDecision(data["route"], str(data.get("reason", ""))[:200])
        except (LLMError, ValueError):
            pass
    if decision is None:
        decision = RouteDecision("both", "default (no classifier)")
    if exact and decision.route in ("vector", "none"):
        decision = RouteDecision("both", f"exact OKF match: {exact.id}")
    return decision
