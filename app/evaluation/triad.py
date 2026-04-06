"""RAG Triad Evaluator — LLM-as-judge for context relevance, groundedness, answer relevance.

All three metrics scored 0.0–1.0.
Uses a dedicated evaluation LLM (settings.EVALUATION_LLM) to avoid self-serving bias.
"""

import asyncio
import re
from typing import List

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.metrics import EVAL_LATENCY, RAG_TRIAD_SCORE
from app.services.llm import get_llm

logger = structlog.get_logger(__name__)

_CONTEXT_RELEVANCE_PROMPT = """You are evaluating a RAG system. Score whether the retrieved context contains information needed to answer the query.

QUERY: {query}

RETRIEVED CONTEXT:
{context}

Scoring criteria:
- 1.0: Context directly answers the query with sufficient detail
- 0.7-0.9: Context contains relevant information but may be incomplete
- 0.4-0.6: Context is partially relevant
- 0.1-0.3: Context is mostly irrelevant
- 0.0: Context is completely irrelevant

Respond with ONLY a float number between 0.0 and 1.0. No explanation."""

_GROUNDEDNESS_PROMPT = """You are evaluating whether an answer is grounded in the provided context.

CONTEXT:
{context}

ANSWER: {answer}

Scoring criteria:
- 1.0: Every claim in the answer is directly supported by the context
- 0.7-0.9: Most claims are supported; minor extrapolation
- 0.4-0.6: Some claims are supported, others are not
- 0.1-0.3: Few claims are grounded; mostly unsupported
- 0.0: Answer contradicts or completely ignores the context

Respond with ONLY a float number between 0.0 and 1.0. No explanation."""

_ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an answer directly addresses the user's question.

QUESTION: {query}

ANSWER: {answer}

Scoring criteria:
- 1.0: Answer directly and completely addresses the question
- 0.7-0.9: Answer mostly addresses the question with minor gaps
- 0.4-0.6: Answer partially addresses the question
- 0.1-0.3: Answer is tangentially related
- 0.0: Answer does not address the question at all

Respond with ONLY a float number between 0.0 and 1.0. No explanation."""


class RAGTriadEvaluator:
    """Evaluates RAG responses using three independent LLM-as-judge metrics."""

    def __init__(self) -> None:
        self._llm = get_llm(model=settings.EVALUATION_LLM)

    async def evaluate_all(
        self, query: str, contexts: List[str], answer: str
    ) -> tuple[float, float, float]:
        """Run all three metrics in parallel. Returns (context_relevance, groundedness, answer_relevance)."""
        import time
        start = time.perf_counter()

        context_str = "\n\n---\n\n".join(contexts)

        cr, gd, ar = await asyncio.gather(
            self.context_relevance(query, context_str),
            self.groundedness(answer, context_str),
            self.answer_relevance(query, answer),
        )

        # Update Prometheus gauges (rolling last value)
        RAG_TRIAD_SCORE.labels(metric="context_relevance").set(cr)
        RAG_TRIAD_SCORE.labels(metric="groundedness").set(gd)
        RAG_TRIAD_SCORE.labels(metric="answer_relevance").set(ar)

        EVAL_LATENCY.labels(metric="all").observe(time.perf_counter() - start)
        logger.info(
            "triad_evaluation_completed",
            context_relevance=cr,
            groundedness=gd,
            answer_relevance=ar,
        )
        return cr, gd, ar

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def context_relevance(self, query: str, context: str) -> float:
        import time
        start = time.perf_counter()
        prompt = _CONTEXT_RELEVANCE_PROMPT.format(query=query, context=context[:4000])
        score = await self._call_judge(prompt)
        EVAL_LATENCY.labels(metric="context_relevance").observe(time.perf_counter() - start)
        return score

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def groundedness(self, answer: str, context: str) -> float:
        import time
        start = time.perf_counter()
        prompt = _GROUNDEDNESS_PROMPT.format(context=context[:4000], answer=answer)
        score = await self._call_judge(prompt)
        EVAL_LATENCY.labels(metric="groundedness").observe(time.perf_counter() - start)
        return score

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def answer_relevance(self, query: str, answer: str) -> float:
        import time
        start = time.perf_counter()
        prompt = _ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
        score = await self._call_judge(prompt)
        EVAL_LATENCY.labels(metric="answer_relevance").observe(time.perf_counter() - start)
        return score

    async def _call_judge(self, prompt: str) -> float:
        """Call the evaluation LLM and parse its float score."""
        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        return self._parse_score(response.content)

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", text.strip())
        if match:
            return max(0.0, min(1.0, float(match.group())))
        logger.warning("eval_score_parse_failed", raw_output=text[:100])
        return 0.5  # neutral fallback
