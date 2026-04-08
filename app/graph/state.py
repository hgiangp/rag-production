"""LangGraph state definitions.

IMPORTANT: Adding a field here requires updating CLAUDE.md Section 4 Rule R14.
All fields must have defaults to allow partial state updates.
"""

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CrossRefTarget(TypedDict, total=False):
    """A resolved cross-reference target extracted from retrieved chunks.

    Populated by the detect_cross_references graph node; consumed by
    fetch_cross_ref_context.  All Qdrant queries use spec_name as the primary
    filter — both intra- and inter-document references flow through the same
    tiered fetch logic (fuzzy spec_name → section_number → semantic fallback).
    """
    spec_name: str           # Qdrant filter — target spec (same doc or different)
    section_number: str      # narrows to a specific clause; absent = spec-level search
    query: str               # semantic search query built from surrounding context


class Citation(TypedDict):
    document_id: str
    filename: str
    chunk_id: str
    score: float
    excerpt: str


class GraphState(TypedDict, total=False):
    """Main graph state — shared across all top-level nodes."""

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_summary: str          # compressed history from summarize_history node
    original_query: str                # raw user message before any rewriting
    rewritten_questions: List[str]     # 1..N decomposed sub-questions

    # ── Routing flags ─────────────────────────────────────────────────────────
    question_is_clear: bool            # False triggers request_clarification interrupt

    # ── Agent results ─────────────────────────────────────────────────────────
    agent_answers: List[str]           # one answer per agent subgraph invocation
    final_answer: str                  # aggregated answer after all agents complete

    # ── RAG context ──────────────────────────────────────────────────────────
    retrieved_contexts: List[str]      # all context chunks used across all agents
    source_citations: List[Citation]   # structured citations for the response

    # ── Request metadata ──────────────────────────────────────────────────────
    user_id: str
    session_id: str
    correlation_id: str

    # ── Self-correction ───────────────────────────────────────────────────────
    self_correction_count: int         # incremented on each re-query cycle


class AgentState(TypedDict, total=False):
    """Agent subgraph state — isolated per sub-question."""

    messages: Annotated[List[BaseMessage], add_messages]
    question: str                      # the sub-question this agent is answering
    context_summary: str               # compressed context from prior tool calls
    answer: str                        # this agent's final answer
    self_correction_count: int
    user_id: str                       # propagated from GraphState for tool collection routing
    correlation_id: str                # propagated for Langfuse span linking inside tools
    cross_ref_targets: List[CrossRefTarget]  # set by detect_cross_references, cleared after fetch
