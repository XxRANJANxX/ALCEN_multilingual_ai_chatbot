# FastAPI Overview

FastAPI is a modern Python web framework for building APIs. It is built on Starlette and Pydantic and uses standard Python type hints to validate requests and generate OpenAPI documentation automatically.

## Async support

FastAPI supports asynchronous endpoints with `async def`, which lets a single worker handle many concurrent requests while waiting on network calls such as LLM APIs. CPU-heavy work, such as computing embeddings, should run in a thread pool so it does not block the event loop.

## Streaming

Responses can be streamed with Server-Sent Events using `StreamingResponse`. Chat applications use streaming to show tokens as they are generated, which lowers the perceived latency.

## Deployment

A common production setup runs Uvicorn workers behind a reverse proxy, packaged in a Docker image. Configuration is usually supplied through environment variables.
