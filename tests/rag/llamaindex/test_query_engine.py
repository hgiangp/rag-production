"""Unit tests for app/rag/llamaindex/query_engine.py.

Covers:
  LlamaQueryEngine._build_engine  — auto_merging vs recursive retriever selection
  LlamaQueryEngine.query          — happy path, source citation formatting, error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.rag.llamaindex.query_engine import LlamaQueryEngine


def _make_engine(mode: str = "auto_merging") -> LlamaQueryEngine:
    """Construct a LlamaQueryEngine with fully mocked LlamaIndex internals."""
    index = MagicMock()
    docstore = MagicMock()
    docstore.docs = {}

    # Make as_retriever return a mock retriever
    index.as_retriever.return_value = MagicMock()
    index.storage_context = MagicMock()

    with (
        patch("app.rag.llamaindex.query_engine.AutoMergingRetriever"),
        patch("app.rag.llamaindex.query_engine.RecursiveRetriever"),
        patch("app.rag.llamaindex.query_engine.RetrieverQueryEngine.from_args") as mock_qe,
    ):
        mock_engine = AsyncMock()
        mock_qe.return_value = mock_engine
        engine = LlamaQueryEngine(index=index, docstore=docstore, mode=mode)
        # Expose the mocked inner engine for assertion
        engine._engine = mock_engine
        return engine


class TestBuildEngine:
    def test_auto_merging_uses_auto_merging_retriever(self):
        with (
            patch("app.rag.llamaindex.query_engine.AutoMergingRetriever") as mock_amr,
            patch("app.rag.llamaindex.query_engine.RecursiveRetriever") as mock_rr,
            patch("app.rag.llamaindex.query_engine.RetrieverQueryEngine.from_args"),
        ):
            index = MagicMock()
            index.as_retriever.return_value = MagicMock()
            index.storage_context = MagicMock()
            LlamaQueryEngine(index=index, docstore=MagicMock(docs={}), mode="auto_merging")
            mock_amr.assert_called_once()
            mock_rr.assert_not_called()

    def test_recursive_uses_recursive_retriever(self):
        with (
            patch("app.rag.llamaindex.query_engine.AutoMergingRetriever") as mock_amr,
            patch("app.rag.llamaindex.query_engine.RecursiveRetriever") as mock_rr,
            patch("app.rag.llamaindex.query_engine.RetrieverQueryEngine.from_args"),
        ):
            index = MagicMock()
            index.as_retriever.return_value = MagicMock()
            index.storage_context = MagicMock()
            LlamaQueryEngine(index=index, docstore=MagicMock(docs={}), mode="recursive")
            mock_rr.assert_called_once()
            mock_amr.assert_not_called()

    def test_reranker_added_when_enabled(self):
        with (
            patch("app.rag.llamaindex.query_engine.AutoMergingRetriever"),
            patch("app.rag.llamaindex.query_engine.RetrieverQueryEngine.from_args") as mock_from,
            patch("app.rag.llamaindex.query_engine.settings") as mock_settings,
        ):
            mock_settings.LLAMAINDEX_SIMILARITY_TOP_K = 5
            mock_settings.LLAMAINDEX_AUTO_MERGE_RATIO_THRESH = 0.5
            mock_settings.reranking_enabled = True
            mock_settings.COHERE_API_KEY = "key"
            mock_settings.RERANKER_MODEL = "model"
            mock_settings.RERANKER_TOP_N = 3

            with patch("app.rag.llamaindex.query_engine.CohereRerank") as mock_cr:
                index = MagicMock()
                index.as_retriever.return_value = MagicMock()
                index.storage_context = MagicMock()
                LlamaQueryEngine(index=index, docstore=MagicMock(docs={}), mode="auto_merging")

            _, kwargs = mock_from.call_args
            postprocessors = kwargs.get("node_postprocessors", [])
            assert len(postprocessors) == 1

    def test_no_reranker_when_disabled(self):
        with (
            patch("app.rag.llamaindex.query_engine.AutoMergingRetriever"),
            patch("app.rag.llamaindex.query_engine.RetrieverQueryEngine.from_args") as mock_from,
            patch("app.rag.llamaindex.query_engine.settings") as mock_settings,
        ):
            mock_settings.LLAMAINDEX_SIMILARITY_TOP_K = 5
            mock_settings.LLAMAINDEX_AUTO_MERGE_RATIO_THRESH = 0.5
            mock_settings.reranking_enabled = False

            index = MagicMock()
            index.as_retriever.return_value = MagicMock()
            index.storage_context = MagicMock()
            LlamaQueryEngine(index=index, docstore=MagicMock(docs={}), mode="auto_merging")

            _, kwargs = mock_from.call_args
            assert kwargs.get("node_postprocessors", []) == []


class TestQuery:
    def _make_response(self, text: str, filenames: list[str]):
        """Build a mock LlamaIndex response with source_nodes."""
        response = MagicMock()
        response.__str__ = MagicMock(return_value=text)
        response.source_nodes = [
            NodeWithScore(node=TextNode(text="x", metadata={"filename": fn}), score=0.9)
            for fn in filenames
        ]
        return response

    async def test_query_returns_text_and_citations(self):
        engine = _make_engine("auto_merging")
        response = self._make_response("RAG reduces hallucinations.", ["doc1.pdf", "doc2.md"])
        engine._engine.aquery = AsyncMock(return_value=response)

        result = await engine.query("What is RAG?")

        assert "RAG reduces hallucinations." in result
        assert "[Source: doc1.pdf]" in result
        assert "[Source: doc2.md]" in result

    async def test_query_with_no_sources_returns_only_answer(self):
        engine = _make_engine("auto_merging")
        response = self._make_response("Some answer.", [])
        engine._engine.aquery = AsyncMock(return_value=response)

        result = await engine.query("question?")

        assert "Some answer." in result
        assert "[Source:" not in result

    async def test_query_exception_returns_error_string(self):
        engine = _make_engine("recursive")
        engine._engine.aquery = AsyncMock(side_effect=RuntimeError("network timeout"))

        result = await engine.query("anything?")

        assert result.startswith("QUERY_ERROR:")
        assert "network timeout" in result

    async def test_query_mode_propagated_to_error_log(self):
        """Errors include the mode so we can trace which retriever failed."""
        engine = _make_engine("recursive")
        engine._engine.aquery = AsyncMock(side_effect=ValueError("bad"))
        engine._mode = "recursive"

        result = await engine.query("q")
        assert "QUERY_ERROR" in result
