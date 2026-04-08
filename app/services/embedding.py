"""Embedding service — provider-agnostic embedding model factory."""

from typing import Any

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Known vector dimensions keyed by model name (lowercase).
# For unlisted models the service probes a live embedding at startup.
_KNOWN_DIMENSIONS: dict[str, int] = {
    # BAAI/BGE family
    "baai/bge-small-en-v1.5": 384,
    "baai/bge-base-en-v1.5": 768,
    "baai/bge-large-en-v1.5": 1024,
    "baai/bge-m3": 1024,
    # sentence-transformers
    "sentence-transformers/all-minilm-l6-v2": 384,
    "sentence-transformers/all-minilm-l12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    # intfloat/E5 family
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    # OpenAI
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


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
                    openai_api_key=settings.OPENAI_EMBEDDING_API_KEY,
                    base_url=settings.EMBEDDING_BASE_URL or None,
                )
            case "ollama":
                from langchain_ollama import OllamaEmbeddings
                return OllamaEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    base_url=settings.OLLAMA_EMBEDDING_BASE_URL,
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
                return OpenAIEmbedding(
                    model=settings.EMBEDDING_MODEL,
                    api_key=settings.OPENAI_EMBEDDING_API_KEY,
                    api_base=settings.EMBEDDING_BASE_URL or None,
                )
            case "ollama":
                from llama_index.embeddings.ollama import OllamaEmbedding
                return OllamaEmbedding(
                    model_name=settings.EMBEDDING_MODEL,
                    base_url=settings.OLLAMA_EMBEDDING_BASE_URL,
                )
            case _:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                return HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)

    def get_vector_size(self) -> int:
        """Return the embedding dimension for the configured model.

        Checks the known-dimensions table first; falls back to a live probe
        so unknown or future models are handled automatically.
        """
        model_key = settings.EMBEDDING_MODEL.lower()
        if model_key in _KNOWN_DIMENSIONS:
            dim = _KNOWN_DIMENSIONS[model_key]
            logger.info("embedding_dimension_from_lookup", model=settings.EMBEDDING_MODEL, dimension=dim)
            return dim

        # Live probe: embed a single token and measure the output length.
        sample = self.model.embed_query("probe")
        dim = len(sample)
        logger.info("embedding_dimension_from_probe", model=settings.EMBEDDING_MODEL, dimension=dim)
        return dim

    async def warmup(self) -> None:
        """Pre-load the embedding model to avoid first-request latency."""
        _ = self.model
        logger.info("embedding_model_warmed_up", model=settings.EMBEDDING_MODEL)


embedding_service = EmbeddingService()
