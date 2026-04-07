"""LlamaIndex hierarchical indexer: Docling layout → heading-based node hierarchy.

Indexing strategy (Strategy B — heading-aware):
  1. Docling HybridChunker → leaf chunks that respect semantic boundaries
     (paragraph / table / list — never cuts mid-sentence or mid-table)
  2. Heading metadata (H1 > H2 > H3) from each chunk defines the parent hierarchy
  3. Parent nodes  = full section text (all leaf texts under that heading concatenated)
     → real document sections, not arbitrary token windows
  4. PARENT / CHILD NodeRelationship links set explicitly on every node
  5. Leaf nodes  → Qdrant (ANN search)
     All nodes   → SimpleDocumentStore (persisted per-collection for auto-merging)

Fallback (flat/no-heading documents):
  HierarchicalNodeParser with token sizes from settings — same behaviour as before,
  but only triggered when Docling finds no heading structure.
"""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
from llama_index.core import Document, Settings as LISettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.metrics import INGEST_CHUNKS

logger = structlog.get_logger(__name__)


class LlamaIndexer:
    """Hierarchical document indexer: Docling layout + heading-based parent hierarchy.

    Call index_document() from the ingest endpoint.
    Call delete_document() from the delete endpoint.
    """

    def __init__(self) -> None:
        self._aclient = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )

    # ── public API ──────────────────────────────────────────────────────────

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

        # Steps 1-3 are CPU-bound → run in thread pool so we don't block the event loop
        leaf_nodes, parent_nodes = await loop.run_in_executor(
            None,
            _build_nodes,
            content, file_type, filename, document_id,
        )

        all_nodes = leaf_nodes + parent_nodes

        # Load or create the per-collection docstore (file I/O → thread pool)
        docstore_path = _docstore_path(collection)
        docstore = await loop.run_in_executor(None, _load_docstore, docstore_path)
        docstore.add_documents(all_nodes)

        # Index leaf nodes into Qdrant using the async client
        vector_store = QdrantVectorStore(
            aclient=self._aclient,
            collection_name=_leaf_collection(collection),
        )
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
        index = VectorStoreIndex([], storage_context=storage_ctx, show_progress=False)
        await index.ainsert_nodes(leaf_nodes)

        # Persist docstore with the new nodes (file I/O → thread pool)
        await loop.run_in_executor(None, docstore.persist, str(docstore_path))

        total = len(all_nodes)
        INGEST_CHUNKS.observe(total)
        logger.info(
            "document_indexed",
            document_id=document_id,
            filename=filename,
            total_nodes=total,
            leaf_nodes=len(leaf_nodes),
            parent_nodes=len(parent_nodes),
        )
        return total

    async def delete_document(self, document_id: str, collection: str) -> None:
        """Remove all nodes for a document from Qdrant and the docstore."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await self._aclient.delete(
            collection_name=_leaf_collection(collection),
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _prune_docstore, _docstore_path(collection), document_id
        )
        logger.info("document_deleted", document_id=document_id, collection=collection)


# ── module-level helpers (all sync — called from thread pool) ───────────────

def _leaf_collection(collection: str) -> str:
    return f"{collection}__llama_leaves"


def _docstore_path(collection: str) -> Path:
    safe = collection.replace("/", "_").replace(":", "_")
    return Path(settings.LLAMAINDEX_DOCSTORE_PATH) / safe


def _build_nodes(
    content: bytes,
    file_type: str,
    filename: str,
    document_id: str,
) -> Tuple[List[TextNode], List[TextNode]]:
    """Parse document and build the node hierarchy. Returns (leaf_nodes, parent_nodes)."""
    # Try Docling-native heading hierarchy first
    try:
        leaf_nodes, parent_nodes = _build_from_docling(content, file_type, filename, document_id)
        if leaf_nodes:
            return leaf_nodes, parent_nodes
        logger.info("docling_no_leaves_fallback", filename=filename)
    except Exception as exc:
        logger.warning("docling_build_failed_fallback", error=str(exc), filename=filename)

    # Fallback: token-based HierarchicalNodeParser
    return _build_from_token_splitter(content, file_type, filename, document_id)


def _build_from_docling(
    content: bytes,
    file_type: str,
    filename: str,
    document_id: str,
) -> Tuple[List[TextNode], List[TextNode]]:
    """Use Docling HybridChunker → heading-based hierarchy.

    Returns (leaf_nodes, parent_nodes) with PARENT/CHILD relationships wired.
    parent_nodes contains one TextNode per unique heading path, whose text is
    the concatenation of all leaf texts under that section — exactly what
    AutoMergingRetriever returns when it merges siblings.
    """
    import os
    import tempfile
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    # Write bytes to a temp file (Docling requires a file path)
    suffix = f".{file_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        docling_doc = result.document
    finally:
        os.unlink(tmp_path)

    # Leaf chunk size = smallest level in the merge hierarchy
    leaf_max_tokens = settings.LLAMAINDEX_AUTO_MERGE_CHUNK_SIZES[-1]
    chunker = HybridChunker(
        tokenizer=settings.EMBEDDING_MODEL,
        max_tokens=leaf_max_tokens,
        merge_peers=True,
    )
    chunks = list(chunker.chunk(docling_doc))

    return _chunks_to_nodes(chunks, document_id, filename)


def _chunks_to_nodes(
    chunks: list,
    document_id: str,
    filename: str,
) -> Tuple[List[TextNode], List[TextNode]]:
    """Convert Docling chunks → LlamaIndex TextNodes with heading-based hierarchy.

    Node structure:
      Leaf: one TextNode per chunk, PARENT → immediate heading section
      Parent: one TextNode per unique heading path, CHILD → [leaves + sub-sections]
               text = concatenation of all descendant leaf texts
    """
    # ── Step 1: collect leaf texts per heading path ──────────────────────────
    # heading_texts[path] = list of leaf texts under that path (and all ancestors)
    heading_texts: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    leaf_entries: List[Tuple[TextNode, Tuple[str, ...]]] = []

    for chunk in chunks:
        raw_headings = getattr(chunk.meta, "headings", None)
        headings: Tuple[str, ...] = tuple(raw_headings) if raw_headings else ()
        text = chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
        if not text:
            continue

        # Accumulate this text at every ancestor heading path
        for level in range(len(headings)):
            heading_texts[headings[: level + 1]].append(text)

        leaf = TextNode(
            text=text,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "node_type": "leaf",
                "heading_path": " > ".join(headings),
            },
        )
        leaf_entries.append((leaf, headings))

    # ── Step 2: create one parent node per unique heading path ───────────────
    heading_to_node: Dict[Tuple[str, ...], TextNode] = {}
    for path, texts in heading_texts.items():
        heading_to_node[path] = TextNode(
            text="\n\n".join(texts),
            metadata={
                "document_id": document_id,
                "filename": filename,
                "node_type": "section",
                "heading": path[-1],
                "heading_level": len(path),
                "heading_path": " > ".join(path),
            },
        )

    # ── Step 3: wire PARENT on section nodes (nested sections) ───────────────
    for path, node in heading_to_node.items():
        if len(path) > 1:
            grandparent = heading_to_node.get(path[:-1])
            if grandparent:
                node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=grandparent.node_id
                )

    # ── Step 4: collect CHILD lists and wire PARENT on leaf nodes ────────────
    children_per_parent: Dict[str, List[RelatedNodeInfo]] = defaultdict(list)

    # Sub-section children (section → sub-section)
    for path, node in heading_to_node.items():
        if len(path) > 1:
            grandparent = heading_to_node.get(path[:-1])
            if grandparent:
                children_per_parent[grandparent.node_id].append(
                    RelatedNodeInfo(node_id=node.node_id)
                )

    # Leaf children (section → leaf)
    all_leaf_nodes: List[TextNode] = []
    for leaf, headings in leaf_entries:
        if headings:
            parent = heading_to_node.get(headings)
            if parent:
                leaf.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent.node_id
                )
                children_per_parent[parent.node_id].append(
                    RelatedNodeInfo(node_id=leaf.node_id)
                )
        all_leaf_nodes.append(leaf)

    # Assign CHILD to every parent so AutoMergingRetriever can count siblings
    for path, node in heading_to_node.items():
        kids = children_per_parent.get(node.node_id)
        if kids:
            node.relationships[NodeRelationship.CHILD] = kids

    return all_leaf_nodes, list(heading_to_node.values())


def _build_from_token_splitter(
    content: bytes,
    file_type: str,
    filename: str,
    document_id: str,
) -> Tuple[List[TextNode], List[TextNode]]:
    """Fallback: token-based HierarchicalNodeParser for flat/no-heading documents."""
    text = _extract_text_basic(content, file_type)
    li_doc = Document(
        text=text,
        metadata={"document_id": document_id, "filename": filename},
    )
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=settings.LLAMAINDEX_AUTO_MERGE_CHUNK_SIZES
    )
    all_nodes = node_parser.get_nodes_from_documents([li_doc])
    leaf_nodes = get_leaf_nodes(all_nodes)
    parent_nodes = [n for n in all_nodes if n not in set(leaf_nodes)]
    return leaf_nodes, parent_nodes


def _extract_text_basic(content: bytes, file_type: str) -> str:
    """Minimal text extraction without Docling (txt / md, or final fallback)."""
    import io
    if file_type == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if file_type == "docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(io.BytesIO(content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return content.decode("utf-8", errors="replace")


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
