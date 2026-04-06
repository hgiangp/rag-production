"""EvalResult database model."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class EvalResultModel(SQLModel, table=True):
    __tablename__ = "eval_result"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    trace_id: str = Field(default="", index=True)
    session_id: Optional[UUID] = Field(default=None, foreign_key="session.id")
    user_id: Optional[UUID] = Field(default=None, foreign_key="user.id")
    query: str
    answer: str
    context_relevance: Optional[float] = None
    groundedness: Optional[float] = None
    answer_relevance: Optional[float] = None
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
