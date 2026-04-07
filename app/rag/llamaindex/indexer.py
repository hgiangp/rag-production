"""LlamaIndex indexer using DoclingReader for document parsing.

Supports two parser modes (configurable via LLAMAINDEX_PARSER_MODE):
  - "markdown": DoclingReader (Markdown export) + MarkdownNodeParser
    → Simpler, good for general use
  - "docling": DoclingReader (JSON export) + DoclingNodeParser
    → Richer grounding info (page numbers, bounding boxes)

Both modes index nodes into Qdrant for vector search.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import List

import structlog
from llama_index.core import Settings as LISettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.metrics import INGEST_CHUNKS

logger = structlog.get_logger(__name__)


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
        )
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
        index = VectorStoreIndex([], storage_context=storage_ctx, show_progress=False)
        await index.ainsert_nodes(nodes)

        # Persist docstore
        await loop.run_in_executor(None, docstore.persist, str(docstore_path))

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


def _parse_with_docling(
    file_path: str,
    document_id: str,
    filename: str,
) -> List[BaseNode]:
    """Use DoclingReader with configured parser mode."""
    mode = settings.LLAMAINDEX_PARSER_MODE.lower()

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
        doc.metadata["document_id"] = document_id
        doc.metadata["filename"] = filename

    # Parse into nodes
    nodes = node_parser.get_nodes_from_documents(documents)

    # Ensure all nodes have the document metadata
    for node in nodes:
        node.metadata["document_id"] = document_id
        node.metadata["filename"] = filename

    logger.info(
        "document_parsed",
        filename=filename,
        document_id=document_id,
        num_documents=len(documents),
        num_nodes=len(nodes),
        parser_mode=mode,
    )

    return nodes


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
    docstore.persist(str(path))
