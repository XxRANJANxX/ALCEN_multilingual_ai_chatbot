from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always look for .env next to requirements.txt, no matter which folder uvicorn is started from.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # utf-8-sig tolerates the invisible BOM that Windows editors/PowerShell add to .env
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8-sig", extra="ignore")

    @field_validator("llm_api_key", "llm_base_url", "llm_model", mode="before")
    @classmethod
    def _clean(cls, v):
        return v.strip().strip('"').strip("'").strip() if isinstance(v, str) else v

    # LLM
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-4o-mini"
    llm_timeout: float = 60.0

    # Embeddings
    embedding_backend: str = "sentence-transformers"  # or "hash"
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Storage
    chroma_dir: str = "./storage/chroma"
    okf_dir: str = "./data/okf"
    docs_dir: str = "./data/docs"
    db_path: str = "./storage/sessions.db"
    eval_dataset: str = "./data/eval/dataset.jsonl"

    # Chunking
    chunk_size: int = 900       # characters
    chunk_overlap: int = 150
    max_upload_mb: int = 20

    # Retrieval
    top_k: int = 5
    min_score: float = 0.80     # cosine threshold for vector hits (model dependent!)
    okf_min_score: float = 0.35
    okf_neighbors: int = 2

    # Conversation
    history_turns: int = 6

    # Behaviour
    enable_verifier: bool = True
    api_key: str = ""           # if set, clients must send X-API-Key
    cors_origins: str = "*"
