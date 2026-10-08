from dataclasses import dataclass, field
from pydantic import BaseModel, Field

@dataclass
class Passage:
    id: str
    text: str
    source: str
    kind: str  # "vector" | "okf"
    score: float
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "score": round(self.score, 3),
            "snippet": self.text[:240],
            "meta": self.meta,
        }


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    language: str | None = Field(default=None, description="Force reply language (ISO code), e.g. 'hi'")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    language: str
    route: str
    route_reason: str
    rewritten_query: str
    grounded: bool
    sources: list[dict]
    verification: dict | None = None
    timings_ms: dict
