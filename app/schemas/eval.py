"""Evaluation schemas."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    session_id: UUID
    query: str
    answer: str
    contexts: List[str] = Field(min_length=1)
    trace_id: Optional[str] = None


class EvalResultResponse(BaseModel):
    id: UUID
    trace_id: str
    query: str
    context_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    groundedness: Optional[float] = Field(None, ge=0.0, le=1.0)
    answer_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    latency_ms: Optional[int] = None
    created_at: datetime


class EvalSummaryResponse(BaseModel):
    total: int
    avg_context_relevance: Optional[float] = None
    avg_groundedness: Optional[float] = None
    avg_answer_relevance: Optional[float] = None
    results: List[EvalResultResponse] = Field(default_factory=list)
