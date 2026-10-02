
import math
import re
import zlib
from abc import ABC, abstractmethod
from app.core.config import Settings


class Embedder(ABC):

    def __init__(self):
        self._qcache: dict[str, list[float]] = {}

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def _embed_query(self, text: str) -> list[float]: ...

    def embed_query(self, text: str) -> list[float]:
        if text not in self._qcache:
            if len(self._qcache) > 256:
                self._qcache.clear()
            self._qcache[text] = self._embed_query(text)
        return self._qcache[text]


class SentenceTransformerEmbedder(Embedder):
    
    def __init__(self, model_name: str):
        super().__init__()

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()

    def embed_documents(self, texts):
        if self.is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def _embed_query(self, text):
        if self.is_e5:
            text = f"query: {text}"
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class HashEmbedder(Embedder):

    def __init__( self, dim: int = 512):
        super().__init__()
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim

        for w in re.findall(r"\w+", text.lower()):
            feats = [w] + [f"#{w}#"[i : i + 3] for i in range(len(w))]
            for f in feats:
                h = zlib.crc32(f.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                v[h % self.dim] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def _embed_query(self, text):
        return self._vec(text)

def make_embedder(s: Settings) -> Embedder:

    if s.embedding_backend == "hash":
        return HashEmbedder()
    
    return SentenceTransformerEmbedder(s.embedding_model)
