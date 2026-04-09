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

import asyncio
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

from app.core.config import settings
from app.core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_RESULTS
from app.graph.state import AgentState
from app.rag.llamaindex.cross_reference import (
    CrossReferenceRetriever,
    DocumentMetadata,
    LLMCrossReferenceExtractor,
)
from app.rag.llamaindex.indexer import _collection_name, _docstore_path
from app.rag.llamaindex.query_engine import LlamaQueryEngine

logger = structlog.get_logger(__name__)

# Cache: (collection, mode) → LlamaQueryEngine
_engines: Dict[Tuple[str, str], LlamaQueryEngine] = {}

# Lazy Langfuse client singleton — used for manual spans inside tools
_langfuse = None


def _get_langfuse():
    global _langfuse
    if not settings.langfuse_enabled:
        return None
    if _langfuse is None:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    return _langfuse


async def _embed_query(text: str) -> List[float]:
    """Embed text via the configured model (runs in executor to avoid blocking event loop)."""
    from app.services.embedding import embedding_service
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedding_service.model.embed_query, text)


_sparse_bm25_model = None


async def _sparse_embed_query(text: str):
    """Generate a BM25 sparse vector via fastembed (same model used at index time).

    Returns a qdrant_client SparseVector suitable for passing directly to Prefetch.query.
    The model is lazy-initialized and reused across calls.
    """
    from fastembed import SparseTextEmbedding
    from qdrant_client.models import SparseVector

    global _sparse_bm25_model
    loop = asyncio.get_event_loop()

    def _run():
        global _sparse_bm25_model
        if _sparse_bm25_model is None:
            _sparse_bm25_model = SparseTextEmbedding("Qdrant/bm25")
        result = next(iter(_sparse_bm25_model.embed([text])))
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    return await loop.run_in_executor(None, _run)


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

        Use for any question requiring document retrieval. Prefer mode='recursive'
        for multi-hop questions that span several sections.

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

            # Retrieve raw nodes — cross-reference detection and resolution is handled
            # by the detect_cross_references + fetch_cross_ref_context graph nodes
            # that run after this tool call completes.
            nodes = await engine.retrieve_nodes(query)

            # Format nodes with enriched source labels so the detect_cross_references
            # node LLM can identify the source spec when classifying intra-doc refs.
            if nodes:
                def _fmt(n) -> str:
                    meta = n.metadata
                    parts = []
                    spec = meta.get("spec_name") or ""
                    if not spec and meta.get("filename"):
                        # Parse from filename convention so the detection LLM always
                        # sees spec= even when the indexer didn't store spec_name.
                        # '7821_(Pop-up)_E_210617.docx' → 'Pop-up'
                        doc_meta = DocumentMetadata.from_filename(meta["filename"])
                        spec = doc_meta.spec_name if doc_meta else ""
                    if spec:
                        parts.append(f"spec={spec}")
                    if meta.get("section_number"):
                        parts.append(f"section={meta['section_number']}")
                    parts.append(f"file={meta.get('filename', 'unknown')}")
                    return f"[Source: {' | '.join(parts)}]\n{n.get_content()}"

                result = "\n---\n".join(_fmt(n) for n in nodes)
            else:
                result = "No relevant content found."

            elapsed = time.perf_counter() - start
            RETRIEVAL_LATENCY.labels(tool="llamaindex_query", collection=collection).observe(elapsed)
            result_count = result.count("[Source:")
            RETRIEVAL_RESULTS.labels(tool="llamaindex_query").observe(result_count)

            logger.info(
                "llamaindex_query_completed",
                mode=mode,
                collection=collection,
                result_length=len(result),
                node_count=len(nodes),
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
          - "(EnlargeWA)" or bare "4.4.1.1" in a Reference table column

        Automatically detects and fetches:
          - Intra-document: sections referenced within the same document
          - Inter-document: chunks from referenced specifications/documents

        Spec names are resolved with fuzzy matching — typos like 'enlagreWA'
        are resolved to the closest stored spec name.

        Args:
            chunk_text:  The full text of the chunk containing references.
            document_id: The document_id of the chunk (from its metadata).

        Returns:
            Resolved context from all detected references, with source labels.
        """
        collection = f"docs_{state['user_id']}"
        start = time.perf_counter()

        try:
            from app.services.llm import get_llm

            # Step 1: LLM-based extraction handles table cells + bare section numbers
            extractor = LLMCrossReferenceExtractor(get_llm(settings.DEFAULT_LLM_MODEL))
            cross_ref = await extractor.extract(chunk_text)

            if cross_ref.is_empty():
                return "No cross-references detected or no matching content found."

            # Derive source spec_name from document_id (filename convention) so that
            # intra-document section refs point at the right spec.
            doc_meta = DocumentMetadata.from_filename(document_id)
            source_spec = doc_meta.spec_name if doc_meta else ""

            targets = []
            for ref in cross_ref.section_refs:
                targets.append({
                    "spec_name": source_spec,
                    "section_number": ref.section_number,
                    "query": ref.original_text,
                })
            for ref in cross_ref.document_refs:
                targets.append({
                    "spec_name": ref.spec_name,
                    "section_number": None,
                    "query": ref.original_text,
                })

            # Step 2: Qdrant fetch with fuzzy spec_name resolution + section fallbacks
            aclient = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY or None,
            )
            retriever = CrossReferenceRetriever(aclient, embed_fn=_embed_query, sparse_embed_fn=_sparse_embed_query)
            contexts = await retriever.resolve_targets(targets, collection)

            elapsed = time.perf_counter() - start
            RETRIEVAL_LATENCY.labels(tool="resolve_cross_references", collection=collection).observe(elapsed)
            RETRIEVAL_RESULTS.labels(tool="resolve_cross_references").observe(len(contexts))

            if not contexts:
                return "No matching content found for the detected cross-references."

            logger.info(
                "explicit_cross_references_resolved",
                collection=collection,
                document_id=document_id,
                ref_count=len(contexts),
                latency=round(elapsed, 3),
            )
            return "\n\n".join(contexts)

        except Exception as exc:
            logger.exception("cross_reference_tool_failed", error=str(exc))
            return f"CROSS_REF_ERROR: {str(exc)}"

    return [llamaindex_query, resolve_cross_references]
