import hashlib
import os

from app.core.config import Settings
from app.core.lang import detect_language
from app.rag.chunker import chunk_text
from app.rag.parsers import UnsupportedFile, parse_file
from app.rag.store import VectorStore, now_iso


class IngestError(Exception):
    pass


def ingest_bytes(store: VectorStore, filename: str, data: bytes, s: Settings) -> dict:

    filename = os.path.basename(filename or "upload")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise IngestError(f"File exceeds {s.max_upload_mb} MB")
    
    try:
        pages = parse_file(filename, data)
    except UnsupportedFile as e:
        raise IngestError(str(e)) from e
    
    except Exception as e:  # corrupt pdf/docx etc.
        raise IngestError(f"Could not parse {filename}: {e}") from e

    doc_id = hashlib.sha1(data).hexdigest()[:12]
    ts = now_iso()
    store.delete_source(filename)  # re-ingesting a file replaces it
    chunks, idx = [], 0

    for page_no, text in pages:
        for ch in chunk_text(text, s.chunk_size, s.chunk_overlap):
            chunks.append({
                "id": f"{doc_id}:{idx}",
                "text": ch.text,
                "metadata": {
                    "source": filename,
                    "doc_id": doc_id,
                    "page": page_no,
                    "section": ch.section,
                    "language": detect_language(ch.text[:400]),
                    "chunk_index": idx,
                    "ingested_at": ts,
                },
            })
            idx += 1
    if not chunks:
        raise IngestError(f"No extractable text in {filename} (scanned PDF? try OCR first)")
    store.add_chunks(chunks)
    
    return {"source": filename, "doc_id": doc_id, "chunks": len(chunks)}
