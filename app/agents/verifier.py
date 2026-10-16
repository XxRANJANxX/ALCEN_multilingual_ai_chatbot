from app.core.jsonutil import parse_json
from app.core.schemas import Passage
from app.llm.client import LLMClient, LLMError

PROMPT = (
    "You are a strict fact-checker. Given numbered context passages and an answer, decide whether "
    "every factual claim in the answer is supported by the passages. Ignore citation markers and "
    "the language of the answer. Reply with JSON only: "
    '{"supported": true|false, "unsupported_claims": ["..."]}'
)


async def verify(llm: LLMClient, answer: str, passages: list[Passage]) -> dict | None:
    ctx = "\n".join(f"[{i}] {p.text[:1200]}" for i, p in enumerate(passages, 1))
    try:
        out = await llm.chat(
            [{"role": "system", "content": PROMPT},
             {"role": "user", "content": f"Passages:\n{ctx}\n\nAnswer:\n{answer}"}],
            temperature=0, max_tokens=400)
        data = parse_json(out)
    except (LLMError, ValueError):
        return None
    return {"supported": bool(data.get("supported", True)),
            "unsupported_claims": [str(c) for c in data.get("unsupported_claims", [])][:5]}
