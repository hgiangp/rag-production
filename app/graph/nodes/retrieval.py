"""Retrieval nodes: token estimation and context compression routing."""

import structlog

from app.core.metrics import GRAPH_NODE_COUNT

logger = structlog.get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def should_compress_context(state: dict) -> dict:
    """No-op node — routing logic is in edges.route_after_compression_check."""
    GRAPH_NODE_COUNT.labels(node="should_compress_context").inc()
    return {}
