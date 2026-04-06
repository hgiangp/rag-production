"""LangGraph tool wrapping LlamaIndex query engine.

Contract (CLAUDE.md Rule R5):
- LlamaIndex is ONLY called through this tool
- Returns formatted string, never raw LlamaIndex objects
"""

import time
from typing import Annotated, List, Literal

import structlog
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_RESULTS
from app.graph.state import AgentState
from app.rag.llamaindex.query_engine import LlamaQueryEngine

logger = structlog.get_logger(__name__)

_engines: dict[str, LlamaQueryEngine] = {}


async def _get_engine(collection: str, mode: str) -> LlamaQueryEngine:
    """Lazy-init query engine per (collection, mode) pair."""
    key = f"{collection}:{mode}"
    if key not in _engines:
        from llama_index.core import Settings as LISettings, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        from app.services.embedding import embedding_service
        from app.services.llm import get_llm

        LISettings.llm = get_llm(model=settings.DEFAULT_LLM_MODEL, framework="llamaindex")
        LISettings.embed_model = embedding_service.llamaindex_model

        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        suffix = "llama_sw" if mode == "sentence_window" else "llama_am"
        coll_name = f"{collection}__{suffix}"

        vector_store = QdrantVectorStore(client=client, collection_name=coll_name)
        index = VectorStoreIndex.from_vector_store(vector_store)
        _engines[key] = LlamaQueryEngine(index=index, mode=mode)

    return _engines[key]


def get_llamaindex_tools() -> List:
    """Build and return all LlamaIndex RAG tools. Collection is resolved per-request from agent state."""

    @tool("llamaindex_query")
    async def llamaindex_query(
        query: str,
        state: Annotated[AgentState, InjectedState],
        mode: Literal["sentence_window", "auto_merging"] = "sentence_window",
    ) -> str:
        """Advanced RAG query using LlamaIndex with sentence-window or auto-merging retrieval.

        Use this when search_child_chunks doesn't return sufficient context,
        or when you need more nuanced retrieval with reranking.

        Args:
            query: The search query
            mode: 'sentence_window' (default) or 'auto_merging'

        Returns:
            Retrieved and reranked context with source references
        """
        collection = f"docs_{state['user_id']}"
        start = time.perf_counter()
        try:
            engine = await _get_engine(collection, mode)
            result = await engine.query(query)

            RETRIEVAL_LATENCY.labels(tool="llamaindex_query", collection=collection).observe(
                time.perf_counter() - start
            )
            logger.info("llamaindex_query_completed", mode=mode, result_length=len(result))
            return result
        except Exception as exc:
            logger.exception("llamaindex_tool_failed", error=str(exc))
            return f"LLAMAINDEX_ERROR: {str(exc)}"

    return [llamaindex_query]
