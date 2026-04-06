"""Embedding service — provider-agnostic embedding model factory."""

from typing import Any

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Lazy-loaded embedding models for LangChain and LlamaIndex."""

    def __init__(self) -> None:
        self._langchain_model = None
        self._llamaindex_model = None

    @property
    def model(self) -> Any:
        """LangChain-compatible embedding model."""
        if self._langchain_model is None:
            self._langchain_model = self._build_langchain()
        return self._langchain_model

    @property
    def llamaindex_model(self) -> Any:
        """LlamaIndex-compatible embedding model."""
        if self._llamaindex_model is None:
            self._llamaindex_model = self._build_llamaindex()
        return self._llamaindex_model

    def _build_langchain(self) -> Any:
        match settings.EMBEDDING_PROVIDER:
            case "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL,
                    encode_kwargs={"batch_size": settings.EMBEDDING_BATCH_SIZE},
                )
            case "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                )
            case "ollama":
                from langchain_ollama import OllamaEmbeddings
                return OllamaEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    base_url=settings.OLLAMA_BASE_URL,
                )
            case _:
                raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")

    def _build_llamaindex(self) -> Any:
        match settings.EMBEDDING_PROVIDER:
            case "huggingface":
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                return HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
            case "openai":
                from llama_index.embeddings.openai import OpenAIEmbedding
                return OpenAIEmbedding(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
            case _:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                return HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)

    async def warmup(self) -> None:
        """Pre-load the embedding model to avoid first-request latency."""
        _ = self.model
        logger.info("embedding_model_warmed_up", model=settings.EMBEDDING_MODEL)


embedding_service = EmbeddingService()
