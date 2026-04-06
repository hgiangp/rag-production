"""Chat endpoints: streaming and non-streaming RAG chat."""

import asyncio
import time
from typing import AsyncGenerator
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.core.auth import CurrentUser, get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.evaluation.runner import EvalRunner
from app.graph.main_graph import get_main_graph
from app.graph.state import GraphState
from app.schemas.chat import ChatRequest, ChatResponse, Citation, Message, StreamChunk, TriadScores

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/invoke", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat_invoke(
    request: Request,
    body: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    """Non-streaming chat: returns full response after pipeline completes."""
    start_ms = int(time.time() * 1000)
    correlation_id = getattr(request.state, "correlation_id", "")
    session_id = body.session_id or current_user.session_id

    graph = await get_main_graph()
    initial_state = GraphState(
        messages=[HumanMessage(content=body.message)],
        user_id=str(current_user.user_id),
        session_id=str(session_id),
        correlation_id=correlation_id,
        conversation_summary="",
        original_query=body.message,
        rewritten_questions=[],
        question_is_clear=True,
        agent_answers=[],
        final_answer="",
        retrieved_contexts=[],
        source_citations=[],
        self_correction_count=0,
    )

    config = {
        "configurable": {"thread_id": f"{current_user.user_id}:{session_id}"},
        "run_name": "rag_chat",
    }

    try:
        result: GraphState = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("graph_invocation_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pipeline failed")

    final_answer = result.get("final_answer", "I could not generate a response.")
    citations = [Citation(**c) for c in result.get("source_citations", [])]
    contexts = result.get("retrieved_contexts", [])
    latency_ms = int(time.time() * 1000) - start_ms

    # Fire-and-forget evaluation (Rule R7)
    if contexts:
        asyncio.create_task(
            EvalRunner().run(
                query=body.message,
                answer=final_answer,
                contexts=contexts,
                session_id=str(session_id),
                user_id=str(current_user.user_id),
                trace_id=correlation_id,
            )
        )

    logger.info(
        "chat_invoke_completed",
        latency_ms=latency_ms,
        citations_count=len(citations),
        answer_length=len(final_answer),
    )

    return ChatResponse(
        session_id=session_id,
        message=Message(role="assistant", content=final_answer),
        citations=citations,
        trace_id=correlation_id,
        latency_ms=latency_ms,
    )


@router.post("/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming chat: yields SSE tokens as they are generated."""
    correlation_id = getattr(request.state, "correlation_id", "")
    session_id = body.session_id or current_user.session_id

    async def _event_stream() -> AsyncGenerator[str, None]:
        graph = await get_main_graph()
        initial_state = GraphState(
            messages=[HumanMessage(content=body.message)],
            user_id=str(current_user.user_id),
            session_id=str(session_id),
            correlation_id=correlation_id,
            conversation_summary="",
            original_query=body.message,
            rewritten_questions=[],
            question_is_clear=True,
            agent_answers=[],
            final_answer="",
            retrieved_contexts=[],
            source_citations=[],
            self_correction_count=0,
        )
        config = {
            "configurable": {"thread_id": f"{current_user.user_id}:{session_id}"},
            "run_name": "rag_chat_stream",
        }

        try:
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        sse = StreamChunk(type="token", content=chunk.content)
                        yield f"data: {sse.model_dump_json()}\n\n"

            # Final done event
            done = StreamChunk(type="done")
            yield f"data: {done.model_dump_json()}\n\n"

        except Exception as exc:
            logger.exception("stream_error", error=str(exc))
            err = StreamChunk(type="error", error=str(exc))
            yield f"data: {err.model_dump_json()}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
