from functools import lru_cache

from app.agents.pipeline import Pipeline
from app.core.config import Settings
from app.llm.client import LLMClient
from app.memory.session import SessionMemory
from app.okf.graph import OKFIndex
from app.rag.embedder import make_embedder
from app.rag.store import VectorStore


class Container:
    """Wires all components together once per process."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = make_embedder(settings)
        self.store = VectorStore(settings.chroma_dir, self.embedder)
        self.okf = OKFIndex(settings.okf_dir, self.embedder)
        self.memory = SessionMemory(settings.db_path)
        self.llm = LLMClient(settings)
        self.pipeline = Pipeline(settings, self.llm, self.store, self.okf, self.memory)


@lru_cache
def get_container() -> Container:
    return Container(Settings())
