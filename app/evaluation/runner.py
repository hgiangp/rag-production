"""Async evaluation runner — fire-and-forget post-response evaluation.

Usage (ALWAYS non-blocking per CLAUDE.md Rule R7):
    asyncio.create_task(eval_runner.run(...))
"""

import time
from typing import List, Optional
from uuid import uuid4

import structlog

from app.core.config import settings
from app.evaluation.schemas import EvalResult
from app.evaluation.triad import RAGTriadEvaluator

logger = structlog.get_logger(__name__)


class EvalRunner:
    """Orchestrates post-response evaluation and persists results."""

    def __init__(self) -> None:
        self._evaluator = RAGTriadEvaluator()

    async def run(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        session_id: str = "",
        user_id: str = "",
        trace_id: str = "",
    ) -> EvalResult:
        """Run triad evaluation and log results to Langfuse + DB.

        This method is designed to be called as a fire-and-forget task.
        """
        start_ms = int(time.time() * 1000)
        result = EvalResult(
            id=uuid4(),
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            query=query,
            answer=answer,
        )

        try:
            cr, gd, ar = await self._evaluator.evaluate_all(
                query=query, contexts=contexts, answer=answer
            )
            result.context_relevance = cr
            result.groundedness = gd
            result.answer_relevance = ar
            result.latency_ms = int(time.time() * 1000) - start_ms

            # Log to Langfuse
            await self._log_to_langfuse(trace_id=trace_id, cr=cr, gd=gd, ar=ar)

            # Persist to DB
            await self._persist(result)

        except Exception as exc:
            logger.exception("eval_runner_failed", trace_id=trace_id, error=str(exc))

        return result

    async def _log_to_langfuse(self, trace_id: str, cr: float, gd: float, ar: float) -> None:
        if not settings.langfuse_enabled or not trace_id:
            return
        try:
            from langfuse import Langfuse
            lf = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            # Normalize to 32-char hex to match how the CallbackHandler stores the trace.
            lf_trace_id = trace_id.replace("-", "")
            lf.score(trace_id=lf_trace_id, name="context_relevance", value=cr)
            lf.score(trace_id=lf_trace_id, name="groundedness", value=gd)
            lf.score(trace_id=lf_trace_id, name="answer_relevance", value=ar)
        except Exception as exc:
            logger.warning("langfuse_score_failed", trace_id=trace_id, error=str(exc))

    async def _persist(self, result: EvalResult) -> None:
        """Write eval result to PostgreSQL."""
        try:
            from sqlmodel.ext.asyncio.session import AsyncSession
            from app.services.database import _async_session_factory

            async with _async_session_factory() as db:
                from app.models.eval_result import EvalResultModel
                row = EvalResultModel(
                    id=result.id,
                    trace_id=result.trace_id,
                    session_id=result.session_id or None,
                    user_id=result.user_id or None,
                    query=result.query,
                    answer=result.answer,
                    context_relevance=result.context_relevance,
                    groundedness=result.groundedness,
                    answer_relevance=result.answer_relevance,
                    latency_ms=result.latency_ms,
                )
                db.add(row)
                await db.commit()
        except Exception as exc:
            logger.warning("eval_persist_failed", error=str(exc))
