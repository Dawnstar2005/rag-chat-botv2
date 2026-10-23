"""Application settings, read once from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    docs_dir: Path  # files placed here are indexed at startup
    frontend_dir: Path
    embeddings_enabled: bool
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_mb: int
    max_history_messages: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", ROOT_DIR / "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Kept separate from DATA_DIR so tests can use a throwaway data dir without re-downloading the model.
    models_dir = Path(os.getenv("MODELS_DIR", ROOT_DIR / "data" / "models")).resolve()
    docs_dir = Path(os.getenv("DOCS_DIR", ROOT_DIR / "documents")).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)  # so the user finds the folder to put files in
    return Settings(
        data_dir=data_dir,
        models_dir=models_dir,
        docs_dir=docs_dir,
        frontend_dir=ROOT_DIR / "frontend",
        embeddings_enabled=_env_bool("EMBEDDINGS_ENABLED", True),
        # Multilingual (~50 languages incl. Mongolian), 384 dims, ~220 MB download on first run.
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        # The embedding model reads only its first 128 tokens (~470 Mongolian characters);
        # longer chunks would be partly invisible to semantic search.
        chunk_size=_env_int("CHUNK_SIZE", 450),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 80),
        top_k=_env_int("TOP_K", 6),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
        max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 20),
    )


settings = load_settings()
