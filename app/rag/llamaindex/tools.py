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
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import structlog
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from llama_index.core import Settings as LISettings, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import PayloadSchemaType

from app.core.config import settings
from app.core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_RESULTS
from app.graph.state import AgentState
from app.rag.llamaindex.cross_reference import CrossReferenceDetector, CrossReferenceRetriever
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

    # No LLM needed — tool returns raw retrieved context; LangGraph orchestrator synthesizes.
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


async def ensure_payload_indexes(collection: str) -> None:
    """Create Qdrant payload indexes for cross-reference filtering.

    Indexes: document_id, spec_name, section_number, model_symbol, language.
    Safe to call multiple times — ignores already-existing indexes.
    """
    aclient = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY or None,
    )
    col = _collection_name(collection)
    fields = {
        "document_id": PayloadSchemaType.KEYWORD,
        "spec_name": PayloadSchemaType.KEYWORD,
        "section_number": PayloadSchemaType.KEYWORD,
        "model_symbol": PayloadSchemaType.KEYWORD,
        "language": PayloadSchemaType.KEYWORD,
    }
    for field, schema in fields.items():
        try:
            await aclient.create_payload_index(
                collection_name=col,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # index already exists
    logger.info("payload_indexes_ensured", collection=col)


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

            # Retrieve raw nodes so we can inspect metadata (document_id, section_number)
            # and auto-resolve any cross-references found in their text.
            nodes = await engine.retrieve_nodes(query)

            # Auto-resolve cross-references transparently — no agent action needed.
            # Uses document_id from node metadata; loop-safe via visited sets.
            extra_contexts: List[str] = []
            detector = CrossReferenceDetector()
            if any(detector.has_references(getattr(n, "text", "")) for n in nodes):
                aclient = AsyncQdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    api_key=settings.QDRANT_API_KEY or None,
                )
                retriever = CrossReferenceRetriever(aclient)
                extra_contexts = await retriever.resolve_from_nodes(nodes, collection)

            # Format retrieved nodes directly — no LLM synthesis here.
            # LangGraph orchestrator handles synthesis from this context.
            if nodes:
                result = "\n---\n".join(
                    f"[Source: {n.metadata.get('filename', 'unknown')}]\n{n.get_content()}"
                    for n in nodes
                )
            else:
                result = "No relevant content found."

            # Append resolved cross-reference context
            if extra_contexts:
                result += "\n\n--- Referenced Context ---\n" + "\n\n".join(extra_contexts)

            elapsed = time.perf_counter() - start
            RETRIEVAL_LATENCY.labels(tool="llamaindex_query", collection=collection).observe(elapsed)
            result_count = result.count("[Source:") + result.count("[Cross-ref:")
            RETRIEVAL_RESULTS.labels(tool="llamaindex_query").observe(result_count)

            logger.info(
                "llamaindex_query_completed",
                mode=mode,
                collection=collection,
                result_length=len(result),
                cross_ref_count=len(extra_contexts),
                latency=round(elapsed, 3),
            )
            return result
        except Exception as exc:
            logger.exception("llamaindex_tool_failed", mode=mode, error=str(exc))
            return f"LLAMAINDEX_ERROR: {str(exc)}"

    @tool("resolve_cross_references")
    async def resolve_cross_references(
        chunk_text: str,
        document_id: str,
        state: Annotated[AgentState, InjectedState],
    ) -> str:
        """Resolve cross-references found inside a retrieved chunk.

        Use this when a retrieved chunk contains phrases like:
          - "see section 3.1.2.3"
          - "refer to spec 'EnlargeWA'"
          - "as defined in 4.2.1"
          - "(EnlargeWA)" — parenthesized spec name

        Automatically detects and fetches:
          - Intra-document: sections referenced within the same document
          - Inter-document: chunks from referenced specifications/documents

        Args:
            chunk_text:  The full text of the chunk containing references.
            document_id: The document_id of the chunk (from its metadata).

        Returns:
            Resolved context from all detected references, with source labels.
        """
        collection = f"docs_{state['user_id']}"
        start = time.perf_counter()

        try:
            aclient = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY or None,
            )
            retriever = CrossReferenceRetriever(aclient)
            extra_contexts = await retriever.resolve_from_text(
                text=chunk_text,
                document_id=document_id,
                collection=collection,
            )

            elapsed = time.perf_counter() - start
            RETRIEVAL_LATENCY.labels(tool="resolve_cross_references", collection=collection).observe(elapsed)
            RETRIEVAL_RESULTS.labels(tool="resolve_cross_references").observe(len(extra_contexts))

            if not extra_contexts:
                return "No cross-references detected or no matching content found."

            logger.info(
                "explicit_cross_references_resolved",
                collection=collection,
                document_id=document_id,
                ref_count=len(extra_contexts),
                latency=round(elapsed, 3),
            )
            return "\n\n".join(extra_contexts)

        except Exception as exc:
            logger.exception("cross_reference_tool_failed", error=str(exc))
            return f"CROSS_REF_ERROR: {str(exc)}"

    return [llamaindex_query, resolve_cross_references]
