"""Evaluation data schemas."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4


@dataclass
class TriadScores:
    context_relevance: Optional[float] = None
    groundedness: Optional[float] = None
    answer_relevance: Optional[float] = None


@dataclass
class EvalResult:
    id: UUID = field(default_factory=uuid4)
    trace_id: str = ""
    session_id: str = ""
    user_id: str = ""
    query: str = ""
    answer: str = ""
    context_relevance: Optional[float] = None
    groundedness: Optional[float] = None
    answer_relevance: Optional[float] = None
    latency_ms: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BatchEvalReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_context_relevance: Optional[float] = None
    avg_groundedness: Optional[float] = None
    avg_answer_relevance: Optional[float] = None
    results: List[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0
