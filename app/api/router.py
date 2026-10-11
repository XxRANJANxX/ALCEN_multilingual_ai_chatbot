import asyncio
import json
import secrets

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.container import Container, get_container
from app.core.schemas import ChatRequest, ChatResponse
from app.eval.runner import run_eval
from app.rag.ingest import IngestError, ingest_bytes


def require_key(x_api_key: str | None = Header(default=None), c: Container = Depends(get_container)):
    expected = c.settings.api_key
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


router = APIRouter(dependencies=[Depends(require_key)])
public = APIRouter()


@public.get("/health")
def health(c: Container = Depends(get_container)):
    return {
        "status": "ok",
        "llm_configured": c.llm.available,
        "llm_model": c.llm.model,
        "embedding_backend": c.settings.embedding_backend,
        "vector_chunks": c.store.count(),
        "okf_nodes": len(c.okf.nodes),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, c: Container = Depends(get_container)):
    return await c.pipeline.chat(req.message, req.session_id, req.language)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, c: Container = Depends(get_container)):
    p = c.pipeline

    async def gen():
        prep = await p.prepare(req.message, req.session_id, req.language)
        yield _sse("meta", p.meta(prep))
        parts: list[str] = []
        async for tok in p.generate_stream(prep):
            parts.append(tok)
            yield _sse("token", {"t": tok})
        answer = "".join(parts)
        p.remember(prep, answer)
        verification = await p.verify(prep, answer)
        yield _sse("done", {"verification": verification, "timings_ms": prep.timings})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.delete("/sessions/{session_id}")
def clear_session(session_id: str, c: Container = Depends(get_container)):
    c.memory.clear(session_id)
    return {"cleared": session_id}


@router.post("/ingest")
async def ingest(files: list[UploadFile] = File(...), c: Container = Depends(get_container)):
    results = []
    for f in files:
        data = await f.read()
        try:
            results.append(await asyncio.to_thread(ingest_bytes, c.store, f.filename or "upload", data, c.settings))
        except IngestError as e:
            results.append({"source": f.filename, "error": str(e)})
    return {"results": results}


@router.get("/sources")
def sources(c: Container = Depends(get_container)):
    return {
        "documents": c.store.sources(),
        "okf": [{"id": n.id, "title": n.title, "type": n.type, "related": n.related}
                for n in c.okf.nodes.values()],
    }


@router.delete("/sources/{name}")
def delete_source(name: str, c: Container = Depends(get_container)):
    n = c.store.delete_source(name)
    if n == 0:
        raise HTTPException(404, f"No source named {name}")
    return {"deleted_chunks": n}


@router.post("/okf/reload")
def okf_reload(c: Container = Depends(get_container)):
    return {"nodes": c.okf.load()}


@router.post("/eval/run")
async def eval_run(generate: bool = False, c: Container = Depends(get_container)):
    return await run_eval(c.pipeline, c.settings.eval_dataset, generate)
