import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import public, router
from app.core.config import Settings

STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_app() -> FastAPI:
    s = Settings()
    
    app = FastAPI(title="Polyglot RAG", version="1.0.0",
                  description="Multilingual RAG + OKF knowledge layer with agentic routing")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(public)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    return app


app = create_app()
