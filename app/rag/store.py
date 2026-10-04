import time
from collections import defaultdict

from app.core.schemas import Passage
from app.rag.embedder import Embedder


class VectorStore:
    def __init__(self, path: str, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks: list[dict]) -> int:
        """chunks: [{id, text, metadata}] - metadata values must be str/int/float/bool."""
        for i in range(0, len(chunks), 128):
            batch = chunks[i : i + 128]
            texts = [c["text"] for c in batch]
            self.col.upsert(
                ids=[c["id"] for c in batch],
                documents=texts,
                embeddings=self.embedder.embed_documents(texts),
                metadatas=[c["metadata"] for c in batch],
            )
        return len(chunks)

    def search(self, query: str, k: int = 10) -> list[Passage]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(Passage(id=cid, text=doc, source=meta.get("source", "?"), kind="vector",
                               score=1.0 - float(dist), meta=dict(meta)))
        return out

    def sources(self) -> list[dict]:
        data = self.col.get(include=["metadatas"])
        agg: dict[str, dict] = defaultdict(lambda: {"chunks": 0})
        for m in data["metadatas"] or []:
            a = agg[m.get("source", "?")]
            a["chunks"] += 1
            a["language"] = m.get("language", "")
            a["ingested_at"] = m.get("ingested_at", "")
        return [{"source": k, **v} for k, v in sorted(agg.items())]

    def delete_source(self, source: str) -> int:
        ids = self.col.get(where={"source": source}, include=[])["ids"]
        if ids:
            self.col.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.col.count()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
