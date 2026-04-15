"""Chat request/response schemas — OpenAI-compatible where possible."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(description="Short excerpt from the source chunk")


class RetrievedChunk(BaseModel):
    """A single retrieved chunk shown to the user during streaming."""
    source: str = Field(description="Source filename")
    content: str = Field(description="Full chunk content")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score if available")
    tool: str = Field(description="Tool that retrieved this chunk")


class WorkflowStep(BaseModel):
    """A single observable step in the RAG pipeline, emitted as a streaming event.

    step_type values:
      query_rewrite     — LLM rewrote and decomposed the user's query
      tool_retrieval    — a retrieval tool returned chunks
      cross_ref_detected — cross-reference targets extracted from retrieved text
      cross_ref_fetched  — cross-reference context fetched from vector store
      aggregation       — N sub-answers merged into the final answer

    status:
      running — step has started, not yet complete (shown as live spinner in UI)
      done    — step completed; data and duration_ms are populated

    data dict shape per step_type (all keys optional for forward compatibility):
      query_rewrite:      {sub_questions: str[], detected_language: str}
      tool_retrieval:     {tool_name: str, chunks_found: int, chunks: RetrievedChunk[]}
      cross_ref_detected: {references: str[], count: int}
      cross_ref_fetched:  {contexts_injected: int}
      aggregation:        {sub_answer_count: int}
    """
    step_type: Literal[
        "query_rewrite",
        "tool_retrieval",
        "cross_ref_detected",
        "cross_ref_fetched",
        "aggregation",
    ]
    title: str
    status: Literal["running", "done"]
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


class TriadScores(BaseModel):
    context_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    groundedness: Optional[float] = Field(None, ge=0.0, le=1.0)
    answer_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[UUID] = None
    stream: bool = False
    collection: Optional[str] = Field(None, description="Override default user collection")
    model: Optional[str] = Field(None, description="Override default LLM model")
    target_language: str = Field(
        "en",
        description=(
            "BCP-47 language code for the answer language (e.g. 'en', 'ja', 'vi', 'fr'). "
            "Defaults to English. The answer will be generated in this language regardless "
            "of the query or document language."
        ),
    )


class ChatResponse(BaseModel):
    session_id: UUID
    message: Message
    citations: List[Citation] = Field(default_factory=list)
    eval_scores: Optional[TriadScores] = None
    trace_id: Optional[str] = None
    latency_ms: Optional[int] = None


class StreamChunk(BaseModel):
    """Single SSE chunk in streaming responses.

    Types:
      token           — LLM answer token
      progress        — pipeline step status (rewrite, retrieve, synthesize)
      retrieved_chunk — chunks returned by a retrieval tool
      citation        — final source citations after generation
      workflow_step   — structured pipeline step (running or done)
      done            — stream completed
      error           — unrecoverable error
    """
    type: Literal["token", "progress", "retrieved_chunk", "citation", "workflow_step", "done", "error"]
    content: str = ""
    citations: Optional[List[Citation]] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    workflow_step: Optional[WorkflowStep] = None
    error: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message: Optional[str] = None


class HistoryMessage(BaseModel):
    id: UUID
    role: str
    content: str
    citations: List[Citation] = Field(default_factory=list)
    eval_scores: Optional[TriadScores] = None
    workflow_steps: List[WorkflowStep] = Field(
        default_factory=list,
        description="Ordered pipeline steps recorded during generation; empty for legacy messages.",
    )
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    session_id: UUID
    name: str
    messages: List[HistoryMessage]


class CreateSessionRequest(BaseModel):
    name: str = Field(default="New Chat", max_length=100)
