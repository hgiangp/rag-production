"""Query processing nodes: summarize history, rewrite query, request clarification."""

import time
from typing import List

import structlog
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.metrics import GRAPH_NODE_COUNT, GRAPH_NODE_LATENCY
from app.graph.state import GraphState

logger = structlog.get_logger(__name__)

# ─── Schemas for structured LLM output ───────────────────────────────────────

class QueryAnalysis(BaseModel):
    questions: List[str] = Field(description="List of clear, standalone sub-questions")
    is_clear: bool = Field(description="Whether the query is clear enough to answer")
    clarification_needed: str = Field(default="", description="What to ask the user if unclear")


# ─── Nodes ────────────────────────────────────────────────────────────────────

async def summarize_history(state: GraphState, llm: BaseChatModel) -> dict:
    """Compress conversation history into a summary to reduce token usage."""
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="summarize_history").inc()

    messages = state.get("messages", [])
    # Need at least 4 messages before summarising
    if len(messages) < 4:
        GRAPH_NODE_LATENCY.labels(node="summarize_history").observe(time.perf_counter() - start)
        return {"conversation_summary": ""}

    # Only consider human/assistant turns (not system messages or tool calls)
    relevant = [
        m for m in messages[:-1]
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    ][-6:]  # last 6 turns

    if not relevant:
        return {"conversation_summary": ""}

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in relevant
    )

    prompt = (
        "Summarize the following conversation history in 2-3 sentences, "
        "focusing on key topics and user intent.\n\n" + history_text
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    summary = resp.content

    logger.info("history_summarized", turn_count=len(relevant), summary_length=len(summary))
    GRAPH_NODE_LATENCY.labels(node="summarize_history").observe(time.perf_counter() - start)
    return {"conversation_summary": summary}


async def rewrite_query(state: GraphState, llm: BaseChatModel) -> dict:
    """Rewrite and optionally decompose the user query into clear sub-questions."""
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="rewrite_query").inc()

    last_msg = state["messages"][-1]
    summary = state.get("conversation_summary", "")

    context = ""
    if summary.strip():
        context = f"Conversation context:\n{summary}\n\n"
    context += f"User query:\n{last_msg.content}"

    system = SystemMessage(content=(
        "You are a query analysis assistant. Given a user query and conversation context:\n"
        "1. Determine if the query is clear and answerable\n"
        "2. If clear, decompose into 1-3 specific, standalone sub-questions\n"
        "3. If unclear, specify exactly what clarification is needed\n"
        "Return structured output."
    ))

    structured_llm = llm.with_structured_output(QueryAnalysis)
    analysis: QueryAnalysis = await structured_llm.ainvoke([system, HumanMessage(content=context)])

    GRAPH_NODE_LATENCY.labels(node="rewrite_query").observe(time.perf_counter() - start)

    if analysis.is_clear and analysis.questions:
        # Clear query: remove old messages to save tokens, keep rewritten questions
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        logger.info("query_rewritten", sub_questions=len(analysis.questions))
        return {
            "question_is_clear": True,
            "messages": delete_all,
            "original_query": last_msg.content,
            "rewritten_questions": analysis.questions,
        }

    # Unclear: append clarification message
    clarification = analysis.clarification_needed or "Could you please provide more details?"
    logger.info("query_needs_clarification")
    return {
        "question_is_clear": False,
        "messages": [AIMessage(content=clarification)],
    }


async def request_clarification(state: GraphState) -> dict:
    """Human-in-the-loop interrupt node — LangGraph pauses here for user input."""
    # This node is registered with interrupt_before=["request_clarification"]
    # The graph will pause here; resume is triggered by the API after user replies
    logger.info("waiting_for_clarification", session_id=state.get("session_id"))
    return {}
