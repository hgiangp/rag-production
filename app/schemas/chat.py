"""Chat request/response schemas — OpenAI-compatible where possible."""

from typing import List, Literal, Optional
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
      done            — stream completed
      error           — unrecoverable error
    """
    type: Literal["token", "progress", "retrieved_chunk", "citation", "done", "error"]
    content: str = ""
    citations: Optional[List[Citation]] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    error: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    session_id: UUID
    messages: List[Message]
