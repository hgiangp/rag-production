"""LlamaIndex indexer using DoclingReader for document parsing.

Supports two parser modes (configurable via LLAMAINDEX_PARSER_MODE):
  - "markdown": DoclingReader (Markdown export) + MarkdownNodeParser
    → Simpler, good for general use
  - "docling": DoclingReader (JSON export) + DoclingNodeParser
    → Richer grounding info (page numbers, bounding boxes)

Both modes index nodes into Qdrant for vector search.

Supports single document (index_document) and batch (index_documents) ingestion.
"""

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import structlog
from llama_index.core import Settings as LISettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from app.rag.llamaindex.cross_reference import DocumentMetadata

from app.core.config import settings
from app.core.metrics import INGEST_CHUNKS
from app.rag.llamaindex.cross_reference import (
    ensure_payload_indexes,
    invalidate_spec_name_cache,
)

logger = structlog.get_logger(__name__)


@dataclass
class DocumentInput:
    """Input for batch document indexing."""

    document_id: str
    filename: str
    content: bytes
    file_type: str


class LlamaIndexer:
    """Document indexer using LlamaIndex DoclingReader.

    Call index_document() from the ingest endpoint.
    Call delete_document() from the delete endpoint.
    """

    def __init__(self) -> None:
        self._aclient = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )

    async def index_document(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        file_type: str,
        collection: str,
    ) -> int:
        """Parse, chunk, embed, and store a document. Returns total node count."""
        from app.services.embedding import embedding_service

        LISettings.embed_model = embedding_service.llamaindex_model

        loop = asyncio.get_event_loop()

        # Parse and chunk in thread pool (CPU-bound)
        nodes = await loop.run_in_executor(
            None,
            _build_nodes,
            content,
            file_type,
            filename,
            document_id,
        )

        if not nodes:
            logger.warning("no_nodes_created", filename=filename, document_id=document_id)
            return 0

        # Load or create the per-collection docstore
        docstore_path = _docstore_path(collection)
        docstore = await loop.run_in_executor(None, _load_docstore, docstore_path)
        docstore.add_documents(nodes)

        # Index nodes into Qdrant
        vector_store = QdrantVectorStore(
            aclient=self._aclient,
            collection_name=_collection_name(collection),
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25"
        )
        
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
        index = VectorStoreIndex([], storage_context=storage_ctx, show_progress=False)
        await index.ainsert_nodes(nodes)

        # Persist docstore (persist expects file path, not directory)
        await loop.run_in_executor(None, docstore.persist, str(docstore_path / "docstore.json"))

        # Ensure Qdrant payload indexes exist for cross-reference filtering fields.
        # Safe to call repeatedly — ignores already-existing indexes.
        # await ensure_payload_indexes(collection)
        # Invalidate spec_name cache so next cross-ref resolution fetches fresh names.
        # invalidate_spec_name_cache(collection)

        total = len(nodes)
        INGEST_CHUNKS.observe(total)
        logger.info(
            "document_indexed",
            document_id=document_id,
            filename=filename,
            total_nodes=total,
            parser_mode=settings.LLAMAINDEX_PARSER_MODE,
        )
        return total

    async def index_documents(
        self,
        documents: List[DocumentInput],
        collection: str,
    ) -> Dict[str, int]:
        """Batch index multiple documents. Returns dict of document_id -> node count.

        More efficient than calling index_document() multiple times:
        - Parses documents in parallel
        - Single docstore load/persist
        - Single Qdrant batch insert
        - No race conditions
        """
        if not documents:
            return {}

        from app.services.embedding import embedding_service

        LISettings.embed_model = embedding_service.llamaindex_model

        loop = asyncio.get_event_loop()

        # Parse all documents in parallel (CPU-bound, use thread pool)
        parse_tasks = [
            loop.run_in_executor(
                None,
                _build_nodes,
                doc.content,
                doc.file_type,
                doc.filename,
                doc.document_id,
            )
            for doc in documents
        ]
        all_node_lists = await asyncio.gather(*parse_tasks)

        # Collect nodes and counts per document
        all_nodes: List[BaseNode] = []
        node_counts: Dict[str, int] = {}
        for doc, nodes in zip(documents, all_node_lists):
            if nodes:
                all_nodes.extend(nodes)
                node_counts[doc.document_id] = len(nodes)
            else:
                node_counts[doc.document_id] = 0
                logger.warning("no_nodes_created", filename=doc.filename, document_id=doc.document_id)

        if not all_nodes:
            logger.warning("batch_index_no_nodes", collection=collection, doc_count=len(documents))
            return node_counts

        # Load docstore once, add all nodes
        docstore_path = _docstore_path(collection)
        docstore = await loop.run_in_executor(None, _load_docstore, docstore_path)
        docstore.add_documents(all_nodes)

        # Index all nodes into Qdrant in one batch
        vector_store = QdrantVectorStore(
            aclient=self._aclient,
            collection_name=_collection_name(collection),
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
        )
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
        index = VectorStoreIndex([], storage_context=storage_ctx, show_progress=False)
        await index.ainsert_nodes(all_nodes)

        # Persist docstore once
        await loop.run_in_executor(None, docstore.persist, str(docstore_path / "docstore.json"))

        await ensure_payload_indexes(collection)
        invalidate_spec_name_cache(collection)

        total = len(all_nodes)
        INGEST_CHUNKS.observe(total)
        logger.info(
            "batch_documents_indexed",
            collection=collection,
            doc_count=len(documents),
            total_nodes=total,
            parser_mode=settings.LLAMAINDEX_PARSER_MODE,
        )
        return node_counts

    async def delete_document(self, document_id: str, collection: str) -> None:
        """Remove all nodes for a document from Qdrant and the docstore."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await self._aclient.delete(
            collection_name=_collection_name(collection),
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _prune_docstore, _docstore_path(collection), document_id
        )
        logger.info("document_deleted", document_id=document_id, collection=collection)

    async def delete_documents(self, document_ids: List[str], collection: str) -> int:
        """Batch delete multiple documents. Returns count of deleted documents.

        More efficient than calling delete_document() multiple times:
        - Single Qdrant delete with OR filter
        - Single docstore prune operation
        """
        if not document_ids:
            return 0

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Delete from Qdrant with OR filter for all document IDs
        await self._aclient.delete(
            collection_name=_collection_name(collection),
            points_selector=Filter(
                should=[
                    FieldCondition(key="document_id", match=MatchValue(value=doc_id))
                    for doc_id in document_ids
                ]
            ),
        )

        # Prune docstore for all documents
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _prune_docstore_batch, _docstore_path(collection), document_ids
        )

        logger.info(
            "batch_documents_deleted",
            collection=collection,
            doc_count=len(document_ids),
        )
        return len(document_ids)


def _collection_name(collection: str) -> str:
    return f"{collection}__llama"


def _docstore_path(collection: str) -> Path:
    safe = collection.replace("/", "_").replace(":", "_")
    return Path(settings.LLAMAINDEX_DOCSTORE_PATH) / safe


def _build_nodes(
    content: bytes,
    file_type: str,
    filename: str,
    document_id: str,
) -> List[BaseNode]:
    """Parse document using DoclingReader and configured node parser."""
    import os

    # Write content to temp file (DoclingReader requires file path)
    suffix = f".{file_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        nodes = _parse_with_docling(tmp_path, document_id, filename)
    finally:
        os.unlink(tmp_path)

    return nodes

def _extract_document_metadata(filename: str) -> Dict: 
    """Add parsed filename fields metadata."""

    doc_meta = DocumentMetadata.from_filename(filename)
    enriched = {}

    enriched["filename"] = filename
    if doc_meta:
        enriched["model_symbol"] = doc_meta.model_symbol
        enriched["spec_name"] = doc_meta.spec_name
        enriched["language"] = doc_meta.language
        enriched["version"] = doc_meta.version
    logger.debug("document_metadata_extracted", filename=filename, metadata=enriched)
    return enriched

def _parse_with_docling(
    file_path: str,
    document_id: str,
    filename: str,
) -> List[BaseNode]:
    """Use DoclingReader with configured parser mode."""
    mode = settings.LLAMAINDEX_PARSER_MODE.lower()
    
    doc_meta = _extract_document_metadata(filename)

    if mode == "docling":
        # JSON export + DoclingNodeParser (richer grounding)
        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        node_parser = DoclingNodeParser()
        logger.debug("using_docling_parser", mode="docling")
    else:
        # Markdown export + MarkdownNodeParser (simpler, default)
        reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
        node_parser = MarkdownNodeParser()
        logger.debug("using_markdown_parser", mode="markdown")

    # Load documents
    documents = reader.load_data(file_path)

    # Add metadata to documents before parsing
    for doc in documents:
        doc.metadata.update(doc_meta)

    # Parse into nodes
    nodes = node_parser.get_nodes_from_documents(documents)

    # Split oversized nodes to fit embedding model context window
    splitter = SentenceSplitter(
        chunk_size=settings.LLAMAINDEX_CHUNK_SIZE,
        chunk_overlap=settings.LLAMAINDEX_CHUNK_OVERLAP,
    )
    final_nodes: List[BaseNode] = []
    for node in nodes:
        if isinstance(node, TextNode) and len(node.text) > settings.LLAMAINDEX_CHUNK_SIZE * 4:
            # Node is likely too large, split it
            split_nodes = splitter.get_nodes_from_documents([node])
            final_nodes.extend(split_nodes)
        else:
            final_nodes.append(node)

    logger.info(
        "document_parsed",
        filename=filename,
        document_id=document_id,
        num_documents=len(documents),
        num_nodes=len(final_nodes),
        nodes_before_split=len(nodes),
        parser_mode=mode,
    )

    return final_nodes


def _load_docstore(path: Path) -> SimpleDocumentStore:
    """Load persisted docstore or create a fresh one."""
    path.mkdir(parents=True, exist_ok=True)
    if (path / "docstore.json").exists():
        return SimpleDocumentStore.from_persist_dir(str(path))
    return SimpleDocumentStore()


def _prune_docstore(path: Path, document_id: str) -> None:
    """Remove nodes belonging to document_id from the persisted docstore."""
    if not (path / "docstore.json").exists():
        return
    docstore = SimpleDocumentStore.from_persist_dir(str(path))
    to_delete = [
        nid for nid, node in docstore.docs.items()
        if getattr(node, "metadata", {}).get("document_id") == document_id
    ]
    for nid in to_delete:
        docstore.delete_document(nid)
    docstore.persist(str(path / "docstore.json"))


def _prune_docstore_batch(path: Path, document_ids: List[str]) -> None:
    """Remove nodes belonging to multiple document_ids from the persisted docstore."""
    if not (path / "docstore.json").exists():
        return
    docstore = SimpleDocumentStore.from_persist_dir(str(path))
    doc_id_set = set(document_ids)
    to_delete = [
        nid for nid, node in docstore.docs.items()
        if getattr(node, "metadata", {}).get("document_id") in doc_id_set
    ]
    for nid in to_delete:
        docstore.delete_document(nid)
    docstore.persist(str(path / "docstore.json"))
