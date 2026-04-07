"""LangGraph tools wrapping LlamaIndex retrieval.

Contract (CLAUDE.md Rule R5):
- LlamaIndex is ONLY called through these tools
- Returns formatted string, never raw LlamaIndex objects

Two tools are exposed to the agent:
  llamaindex_query       — AutoMergingRetriever: merges sibling leaves → parent section
                           when ≥ RATIO_THRESH of a section's children are retrieved.
                           Use for: dense factual questions, table/list lookups.

  llamaindex_recursive   — RecursiveRetriever: walks the heading tree upward from
                           matched leaves to provide broader context.
                           Use for: multi-hop questions that span several sections.
"""

import time
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Tuple

import structlog
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from llama_index.core import Settings as LISettings, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_RESULTS
from app.graph.state import AgentState
from app.rag.llamaindex.indexer import _collection_name, _docstore_path
from app.rag.llamaindex.query_engine import LlamaQueryEngine

logger = structlog.get_logger(__name__)

# Cache: (collection, mode) → LlamaQueryEngine
_engines: Dict[Tuple[str, str], LlamaQueryEngine] = {}


async def _get_engine(collection: str, mode: Literal["auto_merging", "recursive"]) -> LlamaQueryEngine:
    """Lazy-init a LlamaQueryEngine per (collection, mode). Loads docstore from disk."""
    key = (collection, mode)
    if key in _engines:
        return _engines[key]

    from app.services.embedding import embedding_service
    from app.services.llm import get_llm

    LISettings.llm = get_llm(model=settings.DEFAULT_LLM_MODEL, framework="llamaindex")
    LISettings.embed_model = embedding_service.llamaindex_model

    # Load persisted docstore (contains all nodes: leaf + parent sections)
    docstore_dir: Path = _docstore_path(collection)
    if (docstore_dir / "docstore.json").exists():
        docstore = SimpleDocumentStore.from_persist_dir(str(docstore_dir))
    else:
        docstore = SimpleDocumentStore()
        logger.warning("llamaindex_docstore_missing", collection=collection)

    sync_client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY or None,
    )
    vector_store = QdrantVectorStore(
        client=sync_client,
        collection_name=_collection_name(collection),
    )
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)

    engine = LlamaQueryEngine(index=index, docstore=docstore, mode=mode)
    _engines[key] = engine
    logger.info("llamaindex_engine_initialized", collection=collection, mode=mode)
    return engine


def invalidate_engine_cache(collection: str) -> None:
    """Drop cached engines for a collection (call after re-indexing)."""
    for mode in ("auto_merging", "recursive"):
        _engines.pop((collection, mode), None)


def get_llamaindex_tools() -> List:
    """Return all LlamaIndex RAG tools."""

    @tool("llamaindex_query")
    async def llamaindex_query(
        query: str,
        state: Annotated[AgentState, InjectedState],
        mode: Literal["auto_merging", "recursive"] = "auto_merging",
    ) -> str:
        """Hierarchical RAG query using LlamaIndex.

        Retrieves semantically-matched leaf chunks (precision), then merges or
        recurses up the heading hierarchy for richer context (recall).

        Modes:
          auto_merging — merges sibling leaves into their parent section when
            enough siblings match. Best for dense factual questions.
          recursive    — walks up the heading tree from matched leaves.
            Best for multi-hop questions spanning several sections.

        Use this AFTER search_child_chunks returns insufficient or shallow context,
        or for complex questions that require cross-section reasoning.

        Args:
            query: The search query.
            mode:  'auto_merging' (default) or 'recursive'.

        Returns:
            Retrieved context with source citations.
        """
        collection = f"docs_{state['user_id']}"
        start = time.perf_counter()
        try:
            engine = await _get_engine(collection, mode)
            result = await engine.query(query)

            elapsed = time.perf_counter() - start
            RETRIEVAL_LATENCY.labels(tool="llamaindex_query", collection=collection).observe(elapsed)
            # count non-empty source lines as a proxy for result count
            result_count = result.count("[Source:")
            RETRIEVAL_RESULTS.labels(tool="llamaindex_query").observe(result_count)

            logger.info(
                "llamaindex_query_completed",
                mode=mode,
                collection=collection,
                result_length=len(result),
                latency=round(elapsed, 3),
            )
            return result
        except Exception as exc:
            logger.exception("llamaindex_tool_failed", mode=mode, error=str(exc))
            return f"LLAMAINDEX_ERROR: {str(exc)}"

    return [llamaindex_query]
