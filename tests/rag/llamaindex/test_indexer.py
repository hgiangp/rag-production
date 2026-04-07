"""Unit tests for app/rag/llamaindex/indexer.py.

Covers:
  _parse_sync         — Docling path (mocked) and fallback paths
  _chunks_to_nodes    — heading hierarchy, parent/child relationships, metadata
  _build_from_token_splitter — fallback for flat documents
  _build_nodes        — Docling failure falls back gracefully
  _load_docstore      — creates fresh store if dir is empty
  _prune_docstore     — removes nodes for the given document_id
  LlamaIndexer.index_document  — integration (mocked Qdrant + embedding)
  LlamaIndexer.delete_document — integration (mocked Qdrant + docstore prune)
"""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeRelationship, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.rag.llamaindex.indexer import (
    LlamaIndexer,
    _build_from_token_splitter,
    _chunks_to_nodes,
    _load_docstore,
    _prune_docstore,
    _parse_sync,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(text: str, headings: list[str]):
    """Build a minimal Docling-like chunk object."""
    meta = SimpleNamespace(headings=headings)
    return SimpleNamespace(text=text, meta=meta)


# ── _parse_sync ───────────────────────────────────────────────────────────────

class TestParseSync:
    def test_txt_decoded(self):
        content = b"Hello world"
        assert _parse_sync(content, "txt", "file.txt") == "Hello world"

    def test_md_decoded(self):
        content = "# Heading\n\nBody text.".encode()
        result = _parse_sync(content, "md", "file.md")
        assert "Heading" in result

    def test_pdf_fallback_when_docling_absent(self, tmp_path):
        """When docling is not installed, pypdf fallback is used for PDFs."""
        # Build a minimal valid 1-page PDF in memory
        import io
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            buf = io.BytesIO()
            c = rl_canvas.Canvas(buf)
            c.drawString(72, 720, "RAG is useful")
            c.save()
            pdf_bytes = buf.getvalue()
        except ImportError:
            pytest.skip("reportlab not installed — skipping PDF content test")

        with patch.dict("sys.modules", {"docling": None, "docling.document_converter": None}):
            result = _parse_sync(pdf_bytes, "pdf", "test.pdf")
        assert isinstance(result, str)

    def test_docling_exception_falls_back_to_pypdf(self):
        """If Docling raises an unexpected error, pypdf fallback is used."""
        with patch("app.rag.llamaindex.indexer.DocumentConverter", side_effect=RuntimeError("boom")):
            result = _parse_sync(b"%PDF-dummy", "pdf", "test.pdf")
        # pypdf will fail on dummy bytes too, but _parse_sync must not raise
        assert isinstance(result, str)


# ── _chunks_to_nodes ──────────────────────────────────────────────────────────

class TestChunksToNodes:
    def test_returns_leaf_and_parent_nodes(self):
        chunks = [
            _make_chunk("RAG combines retrieval with generation.", ["Introduction"]),
            _make_chunk("It reduces hallucinations.", ["Introduction"]),
            _make_chunk("The retriever fetches passages.", ["Architecture", "Retriever"]),
        ]
        leaves, parents = _chunks_to_nodes(chunks, "doc1", "file.md")
        assert len(leaves) == 3
        # Unique heading paths: ["Introduction"], ["Architecture"], ["Architecture","Retriever"]
        assert len(parents) == 3

    def test_leaf_metadata(self):
        chunks = [_make_chunk("Text A.", ["Section One"])]
        leaves, _ = _chunks_to_nodes(chunks, "doc42", "report.pdf")
        leaf = leaves[0]
        assert leaf.metadata["document_id"] == "doc42"
        assert leaf.metadata["filename"] == "report.pdf"
        assert leaf.metadata["node_type"] == "leaf"
        assert leaf.metadata["heading_path"] == "Section One"

    def test_parent_metadata(self):
        chunks = [_make_chunk("Text B.", ["Chapter 1", "Sub 1.1"])]
        _, parents = _chunks_to_nodes(chunks, "doc1", "book.pdf")
        paths = {p.metadata["heading_path"] for p in parents}
        assert "Chapter 1" in paths
        assert "Chapter 1 > Sub 1.1" in paths

    def test_leaf_has_parent_relationship(self):
        chunks = [_make_chunk("Text.", ["H1", "H2"])]
        leaves, parents = _chunks_to_nodes(chunks, "d", "f.md")
        leaf = leaves[0]
        assert NodeRelationship.PARENT in leaf.relationships

    def test_parent_has_child_relationships(self):
        chunks = [
            _make_chunk("Leaf A.", ["H1"]),
            _make_chunk("Leaf B.", ["H1"]),
        ]
        leaves, parents = _chunks_to_nodes(chunks, "d", "f.md")
        h1_node = next(p for p in parents if p.metadata["heading_path"] == "H1")
        child_rels = h1_node.relationships.get(NodeRelationship.CHILD, [])
        assert len(child_rels) == 2

    def test_nested_section_parent_relationship(self):
        """A sub-section node must point to its parent section."""
        chunks = [
            _make_chunk("Top level text.", ["H1"]),
            _make_chunk("Sub-section text.", ["H1", "H2"]),
        ]
        _, parents = _chunks_to_nodes(chunks, "d", "f.md")
        h1 = next(p for p in parents if p.metadata["heading_path"] == "H1")
        h2 = next(p for p in parents if p.metadata["heading_path"] == "H1 > H2")
        assert NodeRelationship.PARENT in h2.relationships
        assert h2.relationships[NodeRelationship.PARENT].node_id == h1.node_id

    def test_parent_text_contains_all_leaf_texts(self):
        """Parent section text must be the concatenation of all descendant leaf texts."""
        chunks = [
            _make_chunk("First sentence.", ["Intro"]),
            _make_chunk("Second sentence.", ["Intro"]),
        ]
        _, parents = _chunks_to_nodes(chunks, "d", "f.md")
        intro = next(p for p in parents if p.metadata["heading_path"] == "Intro")
        assert "First sentence." in intro.text
        assert "Second sentence." in intro.text

    def test_empty_text_chunks_are_skipped(self):
        chunks = [
            _make_chunk("", ["H1"]),
            _make_chunk("   ", ["H1"]),
            _make_chunk("Real content.", ["H1"]),
        ]
        leaves, _ = _chunks_to_nodes(chunks, "d", "f.md")
        assert len(leaves) == 1

    def test_no_heading_chunks_have_empty_path(self):
        """Chunks without headings get an empty heading_path and no parent."""
        chunks = [_make_chunk("Plain text with no heading.", [])]
        leaves, parents = _chunks_to_nodes(chunks, "d", "f.md")
        assert len(leaves) == 1
        assert leaves[0].metadata["heading_path"] == ""
        assert len(parents) == 0


# ── _build_from_token_splitter ────────────────────────────────────────────────

class TestBuildFromTokenSplitter:
    def test_produces_nodes_for_flat_text(self, flat_bytes):
        leaves, parents = _build_from_token_splitter(flat_bytes, "txt", "flat.txt", "doc99")
        assert len(leaves) > 0
        # HierarchicalNodeParser always creates parent levels
        assert len(parents) > 0

    def test_leaf_metadata_contains_document_id(self, flat_bytes):
        leaves, _ = _build_from_token_splitter(flat_bytes, "txt", "flat.txt", "docX")
        for leaf in leaves:
            assert leaf.metadata.get("document_id") == "docX"


# ── _load_docstore ────────────────────────────────────────────────────────────

class TestLoadDocstore:
    def test_creates_fresh_store_when_dir_empty(self, tmp_path):
        store = _load_docstore(tmp_path / "new_collection")
        assert isinstance(store, SimpleDocumentStore)
        assert len(store.docs) == 0

    def test_loads_persisted_store(self, tmp_path):
        store_path = tmp_path / "col"
        store_path.mkdir()
        # Persist a node
        original = SimpleDocumentStore()
        node = TextNode(text="test", metadata={"document_id": "d1"})
        original.add_documents([node])
        original.persist(str(store_path))

        loaded = _load_docstore(store_path)
        assert node.node_id in loaded.docs


# ── _prune_docstore ───────────────────────────────────────────────────────────

class TestPruneDocstore:
    def test_removes_nodes_for_document(self, tmp_path):
        store_path = tmp_path / "col"
        store_path.mkdir()
        store = SimpleDocumentStore()
        keep = TextNode(text="keep", metadata={"document_id": "doc_A"})
        delete = TextNode(text="delete", metadata={"document_id": "doc_B"})
        store.add_documents([keep, delete])
        store.persist(str(store_path))

        _prune_docstore(store_path, "doc_B")

        reloaded = SimpleDocumentStore.from_persist_dir(str(store_path))
        assert keep.node_id in reloaded.docs
        assert delete.node_id not in reloaded.docs

    def test_noop_when_docstore_missing(self, tmp_path):
        # Should not raise
        _prune_docstore(tmp_path / "nonexistent", "doc_X")


# ── LlamaIndexer (async integration — all I/O mocked) ─────────────────────────

class TestLlamaIndexer:
    """Tests for LlamaIndexer using mocked Qdrant and embedding services."""

    @pytest.fixture
    def indexer(self):
        with patch("app.rag.llamaindex.indexer.AsyncQdrantClient"):
            yield LlamaIndexer()

    @pytest.fixture
    def mock_embed(self):
        embed = MagicMock()
        embed.llamaindex_model = MagicMock()
        with patch("app.services.embedding.embedding_service", embed):
            yield embed

    async def test_index_document_returns_positive_count(
        self, tmp_path, indexer, mock_embed, markdown_bytes
    ):
        with (
            patch("app.core.config.settings.LLAMAINDEX_DOCSTORE_PATH", str(tmp_path)),
            patch("app.rag.llamaindex.indexer.VectorStoreIndex") as mock_vi,
            patch("app.rag.llamaindex.indexer.QdrantVectorStore"),
            patch("app.rag.llamaindex.indexer.StorageContext"),
            patch("app.rag.llamaindex.indexer.INGEST_CHUNKS"),
        ):
            mock_index = AsyncMock()
            mock_vi.return_value = mock_index

            count = await indexer.index_document(
                document_id="doc1",
                filename="test.md",
                content=markdown_bytes,
                file_type="md",
                collection="docs_user1",
            )

        assert count > 0

    async def test_index_document_docling_failure_falls_back(
        self, tmp_path, indexer, mock_embed, flat_bytes
    ):
        """Docling failure during indexing must not raise — fallback is used."""
        with (
            patch("app.core.config.settings.LLAMAINDEX_DOCSTORE_PATH", str(tmp_path)),
            patch("app.rag.llamaindex.indexer.VectorStoreIndex", AsyncMock()),
            patch("app.rag.llamaindex.indexer.QdrantVectorStore"),
            patch("app.rag.llamaindex.indexer.StorageContext"),
            patch("app.rag.llamaindex.indexer.INGEST_CHUNKS"),
            patch(
                "app.rag.llamaindex.indexer._build_from_docling",
                side_effect=RuntimeError("docling crash"),
            ),
        ):
            count = await indexer.index_document(
                document_id="doc2",
                filename="flat.txt",
                content=flat_bytes,
                file_type="txt",
                collection="docs_user2",
            )
        assert count > 0

    async def test_delete_document_calls_qdrant_and_prunes_docstore(
        self, tmp_path, indexer, mock_embed
    ):
        # Seed docstore
        store_path = tmp_path / "docs_user3"
        store_path.mkdir(parents=True)
        store = SimpleDocumentStore()
        node = TextNode(text="to delete", metadata={"document_id": "doc3"})
        store.add_documents([node])
        store.persist(str(store_path))

        with (
            patch("app.core.config.settings.LLAMAINDEX_DOCSTORE_PATH", str(tmp_path)),
            patch.object(indexer._aclient, "delete", new_callable=AsyncMock) as mock_delete,
        ):
            await indexer.delete_document(document_id="doc3", collection="docs_user3")

        mock_delete.assert_awaited_once()
        reloaded = SimpleDocumentStore.from_persist_dir(str(store_path))
        assert node.node_id not in reloaded.docs
