"""Application configuration — single source of truth for all settings.

Reads from .env.{APP_ENV} files. Never call os.getenv() outside this module.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


def _get_env() -> Environment:
    raw = os.getenv("APP_ENV", "development").lower()
    match raw:
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


def _load_env_file() -> Optional[str]:
    env = _get_env()
    base = Path(__file__).parent.parent.parent
    candidates = [
        base / f".env.{env.value}.local",
        base / f".env.{env.value}",
        base / ".env.local",
        base / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(dotenv_path=path)
            return str(path)
    return None


_load_env_file()


def _list(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if not val:
        return default
    val = val.strip("\"'")
    if val.startswith("["):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in val.split(",") if v.strip()]


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


class Settings:
    """All application settings, loaded from environment variables."""

    def __init__(self) -> None:
        self.ENVIRONMENT = _get_env()

        # ── App ──────────────────────────────────────────────────────────────
        self.PROJECT_NAME: str = os.getenv("PROJECT_NAME", "rag-production")
        self.VERSION: str = os.getenv("VERSION", "0.1.0")
        self.API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
        self.DEBUG: bool = _bool("DEBUG", False)
        self.ALLOWED_ORIGINS: List[str] = _list("ALLOWED_ORIGINS", ["*"])

        # ── LLM (Inference) ──────────────────────────────────────────────────
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
        self.DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.DEFAULT_LLM_TEMPERATURE: float = _float("DEFAULT_LLM_TEMPERATURE", 0.2)
        self.MAX_TOKENS: int = _int("MAX_TOKENS", 2000)
        self.MAX_LLM_CALL_RETRIES: int = _int("MAX_LLM_CALL_RETRIES", 3)
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

        # ── LLM (Evaluation) ─────────────────────────────────────────────────
        self.EVALUATION_LLM: str = os.getenv("EVALUATION_LLM", "gpt-4o")
        self.EVALUATION_BASE_URL: str = os.getenv("EVALUATION_BASE_URL", "https://api.openai.com/v1")
        self.EVALUATION_API_KEY: str = os.getenv("EVALUATION_API_KEY", self.OPENAI_API_KEY)
        self.EVALUATION_SLEEP_TIME: int = _int("EVALUATION_SLEEP_TIME", 5)

        # ── Embedding ────────────────────────────────────────────────────────
        self.EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 32)

        # ── Database ─────────────────────────────────────────────────────────
        self.POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT: int = _int("POSTGRES_PORT", 5432)
        self.POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rag_production")
        self.POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "changeme")
        self.POSTGRES_POOL_SIZE: int = _int("POSTGRES_POOL_SIZE", 20)
        self.POSTGRES_MAX_OVERFLOW: int = _int("POSTGRES_MAX_OVERFLOW", 10)

        # ── Qdrant ───────────────────────────────────────────────────────────
        self.QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
        self.QDRANT_PORT: int = _int("QDRANT_PORT", 6333)
        self.QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
        self.QDRANT_GRPC_PORT: int = _int("QDRANT_GRPC_PORT", 6334)
        self.QDRANT_PREFER_GRPC: bool = _bool("QDRANT_PREFER_GRPC", False)

        # ── RAG ──────────────────────────────────────────────────────────────
        self.QDRANT_CHILD_COLLECTION: str = os.getenv("QDRANT_CHILD_COLLECTION", "child_chunks")
        self.QDRANT_PARENT_COLLECTION: str = os.getenv("QDRANT_PARENT_COLLECTION", "parent_store")
        self.CHILD_CHUNK_SIZE: int = _int("CHILD_CHUNK_SIZE", 200)
        self.CHILD_CHUNK_OVERLAP: int = _int("CHILD_CHUNK_OVERLAP", 20)
        self.PARENT_CHUNK_SIZE: int = _int("PARENT_CHUNK_SIZE", 1500)
        self.MAX_RETRIEVAL_K: int = _int("MAX_RETRIEVAL_K", 5)
        self.RETRIEVAL_SCORE_THRESHOLD: float = _float("RETRIEVAL_SCORE_THRESHOLD", 0.70)
        self.MAX_SELF_CORRECTION_ITERATIONS: int = _int("MAX_SELF_CORRECTION_ITERATIONS", 3)
        self.BASE_TOKEN_THRESHOLD: int = _int("BASE_TOKEN_THRESHOLD", 8000)
        self.TOKEN_GROWTH_FACTOR: float = _float("TOKEN_GROWTH_FACTOR", 1.5)
        self.MAX_QUERY_LENGTH: int = _int("MAX_QUERY_LENGTH", 2000)
        self.MAX_FILE_SIZE_MB: int = _int("MAX_FILE_SIZE_MB", 50)
        self.ALLOWED_FILE_TYPES: List[str] = _list("ALLOWED_FILE_TYPES", ["pdf", "docx", "txt", "md"])

        # ── LlamaIndex ───────────────────────────────────────────────────────
        self.LLAMAINDEX_SIMILARITY_TOP_K: int = _int("LLAMAINDEX_SIMILARITY_TOP_K", 5)
        self.LLAMAINDEX_AUTO_MERGE_CHUNK_SIZES: List[int] = [
            int(x) for x in _list("LLAMAINDEX_AUTO_MERGE_CHUNK_SIZES", ["2048", "512", "128"])
        ]
        self.LLAMAINDEX_AUTO_MERGE_RATIO_THRESH: float = _float("LLAMAINDEX_AUTO_MERGE_RATIO_THRESH", 0.5)
        self.LLAMAINDEX_DOCSTORE_PATH: str = os.getenv("LLAMAINDEX_DOCSTORE_PATH", "data/docstore")
        # Parser mode: "markdown" (simpler, MarkdownNodeParser) or "docling" (richer, DoclingNodeParser with JSON)
        self.LLAMAINDEX_PARSER_MODE: str = os.getenv("LLAMAINDEX_PARSER_MODE", "markdown")

        # ── Reranking ────────────────────────────────────────────────────────
        self.COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
        self.RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "rerank-multilingual-v3.0")
        self.RERANKER_TOP_N: int = _int("RERANKER_TOP_N", 3)
        self.ENABLE_RERANKING: bool = _bool("ENABLE_RERANKING", False)

        # ── Long-term Memory ─────────────────────────────────────────────────
        self.MEM0_LLM_MODEL: str = os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini")
        self.MEM0_EMBEDDER_MODEL: str = os.getenv("MEM0_EMBEDDER_MODEL", "text-embedding-3-small")
        self.LONG_TERM_MEMORY_COLLECTION: str = os.getenv("LONG_TERM_MEMORY_COLLECTION", "ltm_vectors")
        self.MAX_MEMORIES_PER_USER: int = _int("MAX_MEMORIES_PER_USER", 50)

        # ── Auth ─────────────────────────────────────────────────────────────
        self.JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-this-in-production-min-32-chars")
        self.JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_DAYS: int = _int("JWT_ACCESS_TOKEN_EXPIRE_DAYS", 30)

        # ── Langfuse ─────────────────────────────────────────────────────────
        self.LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # ── Logging ──────────────────────────────────────────────────────────
        self.LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")

        # ── Rate Limiting ────────────────────────────────────────────────────
        self.RATE_LIMIT_ENDPOINTS: dict = {
            "chat": os.getenv("RATE_LIMIT_CHAT", "30 per minute"),
            "chat_stream": os.getenv("RATE_LIMIT_CHAT_STREAM", "20 per minute"),
            "ingest": os.getenv("RATE_LIMIT_INGEST", "10 per minute"),
            "register": os.getenv("RATE_LIMIT_REGISTER", "10 per hour"),
            "login": os.getenv("RATE_LIMIT_LOGIN", "20 per minute"),
        }

        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment-specific defaults (only if not explicitly set)."""
        overrides = {
            Environment.DEVELOPMENT: {"DEBUG": True, "LOG_LEVEL": "DEBUG", "LOG_FORMAT": "console"},
            Environment.TEST: {"DEBUG": True, "LOG_LEVEL": "DEBUG", "LOG_FORMAT": "console"},
            Environment.STAGING: {"DEBUG": False, "LOG_LEVEL": "INFO", "LOG_FORMAT": "json"},
            Environment.PRODUCTION: {"DEBUG": False, "LOG_LEVEL": "WARNING", "LOG_FORMAT": "json"},
        }
        for attr, val in overrides.get(self.ENVIRONMENT, {}).items():
            if attr.upper() not in os.environ:
                setattr(self, attr, val)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

    @property
    def reranking_enabled(self) -> bool:
        return self.ENABLE_RERANKING and bool(self.COHERE_API_KEY)


settings = Settings()
