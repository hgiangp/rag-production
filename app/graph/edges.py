"""LangGraph conditional edge functions (routing logic).

Each function takes the current state and returns a string routing key.
Keep routing logic minimal — no side effects, no I/O.
"""

from typing import Literal

from app.core.config import settings
from app.graph.state import AgentState, GraphState


def route_after_rewrite(
    state: GraphState,
) -> Literal["request_clarification", "spawn_agents"]:
    """After rewrite_query: clarify if unclear, else run agents."""
    if not state.get("question_is_clear", True):
        return "request_clarification"
    return "spawn_agents"


def route_after_orchestrator(
    state: AgentState,
) -> Literal["tools", "collect_answer", "fallback_response"]:
    """After orchestrator LLM call: route to tools, answer collection, or fallback."""
    last_message = state["messages"][-1]

    # LLM wants to call tools
    if getattr(last_message, "tool_calls", None):
        return "tools"

    # Self-correction limit exceeded → fallback
    if state.get("self_correction_count", 0) >= settings.MAX_SELF_CORRECTION_ITERATIONS:
        return "fallback_response"

    return "collect_answer"


def route_after_compression_check(
    state: AgentState,
) -> Literal["compress_context", "orchestrator"]:
    """After should_compress_context: compress if token budget exceeded."""
    from app.graph.nodes.retrieval import _estimate_tokens

    ctx = state.get("context_summary", "") + " ".join(
        m.content for m in state.get("messages", []) if hasattr(m, "content")
    )
    threshold = settings.BASE_TOKEN_THRESHOLD * (
        settings.TOKEN_GROWTH_FACTOR ** state.get("self_correction_count", 0)
    )
    if _estimate_tokens(ctx) > threshold:
        return "compress_context"
    return "orchestrator"
