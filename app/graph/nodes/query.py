"""Query processing nodes: summarize history, rewrite query, request clarification."""

import time
from typing import Dict, List

import structlog
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.metrics import GRAPH_NODE_COUNT, GRAPH_NODE_LATENCY
from app.graph.state import GraphState

logger = structlog.get_logger(__name__)

# ─── Language helpers ─────────────────────────────────────────────────────────

_LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ja": "Japanese",
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "pt": "Portuguese",
    "th": "Thai",
    "ar": "Arabic",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "id": "Indonesian",
}


def _lang_name(code: str) -> str:
    """Return a human-readable language name for a BCP-47 code."""
    return _LANGUAGE_NAMES.get(code.lower(), code)


# ─── Schemas for structured LLM output ───────────────────────────────────────

class QueryAnalysis(BaseModel):
    questions: List[str] = Field(description="List of clear, standalone sub-questions")
    is_clear: bool = Field(description="Whether the query is clear enough to answer")
    clarification_needed: str = Field(default="", description="What to ask the user if unclear")
    detected_language: str = Field(
        default="en",
        description=(
            "BCP-47 code of the language the user wrote in (e.g. 'en', 'ja', 'vi'). "
            "Detect from the query text itself, not from conversation history."
        ),
    )


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


def _build_rewrite_system(doc_lang: str) -> str:
    """Build the system prompt for query rewriting, language-aware."""
    doc_lang_name = _lang_name(doc_lang)
    cross_lingual_rule = (
        f"5. The document collection is primarily in {doc_lang_name} ({doc_lang}). "
        f"If detected_language differs from '{doc_lang}', rewrite sub-questions in "
        f"{doc_lang_name} — translate the user's intent accurately, preserving technical "
        f"terms in {doc_lang_name} — so embedding and keyword search match the actual "
        f"document text. Do NOT translate the clarification_needed field."
    )
    return (
        "You are a query analysis assistant. Given a user query and conversation context:\n"
        "1. Determine if the query is clear and answerable\n"
        "2. If clear, decompose into 1-3 specific, standalone sub-questions\n"
        "3. If unclear, specify exactly what clarification is needed\n"
        "4. Detect the language of the user query and set detected_language (BCP-47 code)\n"
        + cross_lingual_rule + "\n"
        "Return structured output."
    )


async def rewrite_query(state: GraphState, llm: BaseChatModel) -> dict:
    """Rewrite and optionally decompose the user query into clear sub-questions.

    Also detects the query language and, when query language differs from the
    configured document language, rewrites sub-questions in the document language
    so retrieval (embedding + BM25) matches actual document text.
    """
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="rewrite_query").inc()

    last_msg = state["messages"][-1]
    summary = state.get("conversation_summary", "")

    context = ""
    if summary.strip():
        context = f"Conversation context:\n{summary}\n\n"
    context += f"User query:\n{last_msg.content}"

    doc_lang = settings.DOCUMENT_LANGUAGE
    system = SystemMessage(content=_build_rewrite_system(doc_lang))

    structured_llm = llm.with_structured_output(QueryAnalysis)
    analysis: QueryAnalysis = await structured_llm.ainvoke([system, HumanMessage(content=context)])

    detected = analysis.detected_language or "en"
    GRAPH_NODE_LATENCY.labels(node="rewrite_query").observe(time.perf_counter() - start)

    if analysis.is_clear:
        # Clear query: remove old messages to save tokens, keep rewritten questions.
        # Fall back to original query if the LLM returned an empty questions list.
        questions = analysis.questions if analysis.questions else [last_msg.content]
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        logger.info(
            "query_rewritten",
            sub_questions=len(questions),
            query_language=detected,
            doc_language=doc_lang,
            cross_lingual=(detected != doc_lang),
        )
        return {
            "question_is_clear": True,
            "messages": delete_all,
            "original_query": last_msg.content,
            "rewritten_questions": questions,
            "query_language": detected,
        }

    # Unclear: append clarification message
    clarification = analysis.clarification_needed or "Could you please provide more details?"
    logger.info("query_needs_clarification", query_language=detected)
    return {
        "question_is_clear": False,
        "query_language": detected,
        "messages": [AIMessage(content=clarification)],
    }


async def request_clarification(state: GraphState) -> dict:
    """Human-in-the-loop interrupt node — LangGraph pauses here for user input."""
    # This node is registered with interrupt_before=["request_clarification"]
    # The graph will pause here; resume is triggered by the API after user replies
    logger.info("waiting_for_clarification", session_id=state.get("session_id"))
    return {}
