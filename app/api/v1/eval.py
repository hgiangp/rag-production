"""Evaluation endpoints: trigger eval and fetch results."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Request
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import CurrentUser, get_current_user
from app.core.limiter import limiter
from app.evaluation.runner import EvalRunner
from app.evaluation.schemas import EvalResult
from app.schemas.eval import EvalRequest, EvalResultResponse, EvalSummaryResponse
from app.services.database import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])

_runner = EvalRunner()


@router.post("/run", response_model=EvalResultResponse)
@limiter.limit("20/minute")
async def run_evaluation(
    request: Request,
    body: EvalRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> EvalResultResponse:
    """Trigger synchronous evaluation for a specific query/answer pair."""
    result = await _runner.run(
        query=body.query,
        answer=body.answer,
        contexts=body.contexts,
        session_id=str(body.session_id),
        user_id=str(current_user.user_id),
        trace_id=body.trace_id or "",
    )
    return EvalResultResponse(
        id=result.id,
        trace_id=result.trace_id,
        query=result.query,
        context_relevance=result.context_relevance,
        groundedness=result.groundedness,
        answer_relevance=result.answer_relevance,
        latency_ms=result.latency_ms,
        created_at=result.created_at,
    )


@router.get("/results", response_model=EvalSummaryResponse)
async def get_eval_results(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    limit: int = 20,
) -> EvalSummaryResponse:
    """Fetch recent evaluation results for the current user."""
    from app.models.eval_result import EvalResultModel

    results = await db.exec(
        select(EvalResultModel)
        .where(EvalResultModel.user_id == current_user.user_id)
        .order_by(EvalResultModel.created_at.desc())
        .limit(limit)
    )
    rows = results.all()

    if not rows:
        return EvalSummaryResponse(total=0)

    def _avg(field: str) -> Optional[float]:
        vals = [getattr(r, field) for r in rows if getattr(r, field) is not None]
        return sum(vals) / len(vals) if vals else None

    return EvalSummaryResponse(
        total=len(rows),
        avg_context_relevance=_avg("context_relevance"),
        avg_groundedness=_avg("groundedness"),
        avg_answer_relevance=_avg("answer_relevance"),
        results=[
            EvalResultResponse(
                id=r.id,
                trace_id=r.trace_id,
                query=r.query,
                context_relevance=r.context_relevance,
                groundedness=r.groundedness,
                answer_relevance=r.answer_relevance,
                latency_ms=r.latency_ms,
                created_at=r.created_at,
            )
            for r in rows
        ],
    )
