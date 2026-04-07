"""Unit tests for app/rag/llamaindex/indexer.py.

Covers:
  _build_nodes          — DoclingReader parsing with both parser modes
  _parse_with_docling   — Markdown and Docling parser modes
  _load_docstore        — creates fresh store if dir is empty
  _prune_docstore       — removes nodes for the given document_id
  LlamaIndexer.index_document  — integration (mocked Qdrant + embedding)
  LlamaIndexer.delete_document — integration (mocked Qdrant + docstore prune)
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.rag.llamaindex.indexer import (
    LlamaIndexer,
    _build_nodes,
    _load_docstore,
    _prune_docstore,
)


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
        original.persist(str(store_path / "docstore.json"))

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
        store.persist(str(store_path / "docstore.json"))

        _prune_docstore(store_path, "doc_B")

        reloaded = SimpleDocumentStore.from_persist_dir(str(store_path))
        assert keep.node_id in reloaded.docs
        assert delete.node_id not in reloaded.docs

    def test_noop_when_docstore_missing(self, tmp_path):
        # Should not raise
        _prune_docstore(tmp_path / "nonexistent", "doc_X")


# ── _build_nodes ──────────────────────────────────────────────────────────────

class TestBuildNodes:
    def test_builds_nodes_from_markdown_content(self, markdown_bytes):
        """Markdown content should produce nodes."""
        with patch("app.core.config.settings.LLAMAINDEX_PARSER_MODE", "markdown"):
            nodes = _build_nodes(markdown_bytes, "md", "test.md", "doc1")
        assert len(nodes) > 0
        # Check metadata is set
        for node in nodes:
            assert node.metadata.get("document_id") == "doc1"
            assert node.metadata.get("filename") == "test.md"

    def test_builds_nodes_from_txt_content(self, flat_bytes):
        """Plain text content should produce nodes."""
        with patch("app.core.config.settings.LLAMAINDEX_PARSER_MODE", "markdown"):
            nodes = _build_nodes(flat_bytes, "txt", "test.txt", "doc2")
        assert len(nodes) > 0

    def test_docling_parser_mode(self, markdown_bytes):
        """Docling parser mode should also produce nodes."""
        with patch("app.core.config.settings.LLAMAINDEX_PARSER_MODE", "docling"):
            nodes = _build_nodes(markdown_bytes, "md", "test.md", "doc3")
        assert len(nodes) > 0
        for node in nodes:
            assert node.metadata.get("document_id") == "doc3"


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
            patch("app.core.config.settings.LLAMAINDEX_PARSER_MODE", "markdown"),
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

    async def test_index_document_with_docling_mode(
        self, tmp_path, indexer, mock_embed, markdown_bytes
    ):
        """Test indexing with docling parser mode."""
        with (
            patch("app.core.config.settings.LLAMAINDEX_DOCSTORE_PATH", str(tmp_path)),
            patch("app.core.config.settings.LLAMAINDEX_PARSER_MODE", "docling"),
            patch("app.rag.llamaindex.indexer.VectorStoreIndex") as mock_vi,
            patch("app.rag.llamaindex.indexer.QdrantVectorStore"),
            patch("app.rag.llamaindex.indexer.StorageContext"),
            patch("app.rag.llamaindex.indexer.INGEST_CHUNKS"),
        ):
            mock_index = AsyncMock()
            mock_vi.return_value = mock_index

            count = await indexer.index_document(
                document_id="doc2",
                filename="test.md",
                content=markdown_bytes,
                file_type="md",
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
        store.persist(str(store_path / "docstore.json"))

        with (
            patch("app.core.config.settings.LLAMAINDEX_DOCSTORE_PATH", str(tmp_path)),
            patch.object(indexer._aclient, "delete", new_callable=AsyncMock) as mock_delete,
        ):
            await indexer.delete_document(document_id="doc3", collection="docs_user3")

        mock_delete.assert_awaited_once()
        reloaded = SimpleDocumentStore.from_persist_dir(str(store_path))
        assert node.node_id not in reloaded.docs
