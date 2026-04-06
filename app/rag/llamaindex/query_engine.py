"""LlamaIndex query engine with optional reranking and metadata filtering."""

from typing import Literal, Optional

import structlog
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LlamaQueryEngine:
    """Wraps LlamaIndex retrieval with reranking and context window expansion."""

    def __init__(self, index: VectorStoreIndex, mode: Literal["sentence_window", "auto_merging"] = "sentence_window") -> None:
        self._index = index
        self._mode = mode
        self._engine = self._build_engine()

    def _build_engine(self) -> RetrieverQueryEngine:
        retriever = self._index.as_retriever(similarity_top_k=settings.LLAMAINDEX_SIMILARITY_TOP_K)
        postprocessors = []

        if self._mode == "sentence_window":
            postprocessors.append(
                MetadataReplacementPostProcessor(target_metadata_key="window")
            )

        if settings.reranking_enabled:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
            postprocessors.append(
                CohereRerank(
                    api_key=settings.COHERE_API_KEY,
                    model=settings.RERANKER_MODEL,
                    top_n=settings.RERANKER_TOP_N,
                )
            )

        return self._index.as_query_engine(
            node_postprocessors=postprocessors,
            similarity_top_k=settings.LLAMAINDEX_SIMILARITY_TOP_K,
        )

    async def query(self, query: str) -> str:
        """Execute a RAG query and return formatted results."""
        try:
            response = await self._engine.aquery(query)
            source_info = "\n".join(
                f"[Source: {n.metadata.get('filename', 'unknown')}]"
                for n in getattr(response, "source_nodes", [])
            )
            result = str(response)
            if source_info:
                result += f"\n\n{source_info}"
            return result
        except Exception as exc:
            logger.exception("llamaindex_query_failed", error=str(exc))
            return f"QUERY_ERROR: {str(exc)}"
