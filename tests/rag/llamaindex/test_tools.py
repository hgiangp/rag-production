"""Unit tests for app/rag/llamaindex/tools.py.

Covers:
  get_llamaindex_tools       — tool is a callable with the right name
  llamaindex_query (tool)    — happy path, both modes, error handling
  _get_engine                — lazy init, docstore loading, cache behaviour
  invalidate_engine_cache    — drops cached entries for the collection
"""

from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.rag.llamaindex.tools as tools_module
from app.rag.llamaindex.tools import get_llamaindex_tools, invalidate_engine_cache


# ── helpers ──────────────────────────────────────────────────────────────────

def _agent_state(user_id: str = "user_test") -> dict:
    return {"user_id": user_id, "question": "What is RAG?"}


def _mock_engine(result: str = "RAG answer. [Source: doc.pdf]") -> MagicMock:
    engine = AsyncMock()
    engine.query = AsyncMock(return_value=result)
    return engine


# ── tool registration ─────────────────────────────────────────────────────────

class TestGetLlamaindexTools:
    def test_returns_non_empty_list(self):
        tools = get_llamaindex_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tool_named_llamaindex_query(self):
        tools = get_llamaindex_tools()
        names = [t.name for t in tools]
        assert "llamaindex_query" in names

    def test_tool_is_callable(self):
        tools = get_llamaindex_tools()
        tool = next(t for t in tools if t.name == "llamaindex_query")
        assert callable(tool.func) or callable(getattr(tool, "coroutine", None)) or callable(tool)


# ── _get_engine cache behaviour ───────────────────────────────────────────────

class TestGetEngineCache:
    def setup_method(self):
        tools_module._engines.clear()

    async def test_engine_is_cached_after_first_call(self, tmp_path):
        with (
            patch("app.rag.llamaindex.tools.settings") as ms,
            patch("app.rag.llamaindex.tools.QdrantClient"),
            patch("app.rag.llamaindex.tools.QdrantVectorStore"),
            patch("app.rag.llamaindex.tools.StorageContext"),
            patch("app.rag.llamaindex.tools.VectorStoreIndex"),
            patch("app.rag.llamaindex.tools.LlamaQueryEngine") as mock_lqe,
            patch("app.rag.llamaindex.tools.SimpleDocumentStore") as mock_ds,
            patch("app.rag.llamaindex.tools._docstore_path", return_value=tmp_path / "col"),
            patch("app.services.embedding.embedding_service"),
            patch("app.services.llm.get_llm"),
        ):
            ms.DEFAULT_LLM_MODEL = "gpt-4o-mini"
            ms.QDRANT_HOST = "localhost"
            ms.QDRANT_PORT = 6333
            ms.QDRANT_API_KEY = ""

            mock_lqe.return_value = MagicMock()
            mock_ds.return_value = MagicMock(docs={})

            from app.rag.llamaindex.tools import _get_engine
            e1 = await _get_engine("docs_u1", "auto_merging")
            e2 = await _get_engine("docs_u1", "auto_merging")

        assert e1 is e2
        assert mock_lqe.call_count == 1

    async def test_different_modes_get_different_engines(self, tmp_path):
        with (
            patch("app.rag.llamaindex.tools.settings") as ms,
            patch("app.rag.llamaindex.tools.QdrantClient"),
            patch("app.rag.llamaindex.tools.QdrantVectorStore"),
            patch("app.rag.llamaindex.tools.StorageContext"),
            patch("app.rag.llamaindex.tools.VectorStoreIndex"),
            patch("app.rag.llamaindex.tools.LlamaQueryEngine") as mock_lqe,
            patch("app.rag.llamaindex.tools.SimpleDocumentStore") as mock_ds,
            patch("app.rag.llamaindex.tools._docstore_path", return_value=tmp_path / "col"),
            patch("app.services.embedding.embedding_service"),
            patch("app.services.llm.get_llm"),
        ):
            ms.DEFAULT_LLM_MODEL = "gpt-4o-mini"
            ms.QDRANT_HOST = "localhost"
            ms.QDRANT_PORT = 6333
            ms.QDRANT_API_KEY = ""
            mock_lqe.side_effect = [MagicMock(), MagicMock()]
            mock_ds.return_value = MagicMock(docs={})

            from app.rag.llamaindex.tools import _get_engine
            e_am = await _get_engine("docs_u2", "auto_merging")
            e_re = await _get_engine("docs_u2", "recursive")

        assert e_am is not e_re
        assert mock_lqe.call_count == 2


# ── invalidate_engine_cache ───────────────────────────────────────────────────

class TestInvalidateEngineCache:
    def test_clears_both_modes(self):
        tools_module._engines[("docs_u", "auto_merging")] = MagicMock()
        tools_module._engines[("docs_u", "recursive")] = MagicMock()
        tools_module._engines[("docs_other", "auto_merging")] = MagicMock()

        invalidate_engine_cache("docs_u")

        assert ("docs_u", "auto_merging") not in tools_module._engines
        assert ("docs_u", "recursive") not in tools_module._engines
        # Other collection is untouched
        assert ("docs_other", "auto_merging") in tools_module._engines

    def test_no_error_when_collection_not_cached(self):
        invalidate_engine_cache("does_not_exist")  # must not raise


# ── llamaindex_query tool behaviour ──────────────────────────────────────────

class TestLlamaindexQueryTool:
    """Call the inner async function directly, bypassing LangChain tool wrapping."""

    def setup_method(self):
        tools_module._engines.clear()

    async def _invoke_tool(self, query: str, mode: str = "auto_merging", user_id: str = "u1") -> str:
        state = _agent_state(user_id)
        engine = _mock_engine()
        tools_module._engines[(f"docs_{user_id}", mode)] = engine

        # Grab the raw coroutine from the tool
        tool_obj = next(t for t in get_llamaindex_tools() if t.name == "llamaindex_query")
        # LangChain tools expose the underlying coroutine via .coroutine or .func
        func = getattr(tool_obj, "coroutine", None) or tool_obj.func
        return await func(query=query, state=state, mode=mode)

    async def test_happy_path_auto_merging(self):
        result = await self._invoke_tool("What is RAG?", mode="auto_merging")
        assert "RAG answer" in result

    async def test_happy_path_recursive(self):
        result = await self._invoke_tool("Explain indexing", mode="recursive")
        assert "RAG answer" in result

    async def test_engine_error_returns_error_string(self):
        user_id = "err_user"
        state = _agent_state(user_id)
        bad_engine = AsyncMock()
        bad_engine.query = AsyncMock(side_effect=RuntimeError("connection refused"))
        tools_module._engines[("docs_err_user", "auto_merging")] = bad_engine

        tool_obj = next(t for t in get_llamaindex_tools() if t.name == "llamaindex_query")
        func = getattr(tool_obj, "coroutine", None) or tool_obj.func
        result = await func(query="q", state=state, mode="auto_merging")

        assert "LLAMAINDEX_ERROR" in result
        assert "connection refused" in result

    async def test_metrics_observed(self):
        with (
            patch("app.rag.llamaindex.tools.RETRIEVAL_LATENCY") as mock_lat,
            patch("app.rag.llamaindex.tools.RETRIEVAL_RESULTS") as mock_res,
        ):
            mock_lat.labels.return_value = MagicMock(observe=MagicMock())
            mock_res.labels.return_value = MagicMock(observe=MagicMock())

            await self._invoke_tool("q")

        mock_lat.labels.assert_called_once()
        mock_res.labels.assert_called_once()
