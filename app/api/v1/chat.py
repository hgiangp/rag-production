"""Chat endpoints: streaming and non-streaming RAG chat."""

import asyncio
import re
import time
from typing import AsyncGenerator, List
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.core.auth import CurrentUser, get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.evaluation.runner import EvalRunner
from app.graph.main_graph import get_main_graph
from app.graph.state import GraphState
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    Message,
    RetrievedChunk,
    StreamChunk,
    TriadScores,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# ── Streaming helpers ──────────────────────────────────────────────────────────

# Human-readable progress messages for graph nodes and tools.
_NODE_PROGRESS = {
    "rewrite_query": "Rewriting and analyzing query...",
    "agent": "Starting retrieval agent...",
    "aggregate_answers": "Synthesizing final answer...",
    "load_user_memory": "Loading conversation memory...",
}
_TOOL_PROGRESS = {
    "search_child_chunks": "Searching document chunks...",
    "fetch_parent_chunks": "Fetching parent context...",
    "llamaindex_query": "Querying documents...",
    "llamaindex_recursive": "Querying documents (recursive)...",
    "resolve_cross_references": "Resolving cross-references...",
}
# Tools whose output should be parsed and shown as retrieved chunks.
_RETRIEVAL_TOOLS = frozenset(_TOOL_PROGRESS.keys())
# Prefixes that signal an error / empty result — don't parse these as chunks.
_ERROR_PREFIXES = (
    "NO_RELEVANT_CHUNKS",
    "NO_PARENT_DOCUMENTS",
    "RETRIEVAL_ERROR",
    "PARENT_RETRIEVAL_ERROR",
    "LLAMAINDEX_ERROR",
    "CROSS_REF_ERROR",
    "No relevant content",
    "No cross-references",
)


def _parse_langchain_chunks(output: str, tool: str) -> List[RetrievedChunk]:
    """Parse search_child_chunks / fetch_parent_chunks output into RetrievedChunk list."""
    chunks: List[RetrievedChunk] = []
    # Both tools emit blocks starting with "Parent ID:" separated by "\n\n" or "\n\n---\n\n".
    parts = re.split(r"\n\n---\n\n|\n\n(?=Parent ID:)", output)
    for part in parts:
        if not part.strip():
            continue
        source_m = re.search(r"^Source:\s*(.+)", part, re.MULTILINE)
        content_m = re.search(r"^Content:\s*(.+)", part, re.DOTALL | re.MULTILINE)
        score_m = re.search(r"^Relevance:\s*([\d.]+)", part, re.MULTILINE)
        if content_m:
            chunks.append(RetrievedChunk(
                source=source_m.group(1).strip() if source_m else "unknown",
                content=content_m.group(1).strip(),
                score=float(score_m.group(1)) if score_m else None,
                tool=tool,
            ))
    return chunks


def _parse_llamaindex_chunks(output: str, tool: str) -> List[RetrievedChunk]:
    """Parse llamaindex_query / llamaindex_recursive output into RetrievedChunk list."""
    chunks: List[RetrievedChunk] = []
    # Format: "[Source: filename]\ncontent\n---\n[Source: filename2]\ncontent2"
    parts = output.split("\n---\n")
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        source_m = re.search(r"\[Source:\s*(.+?)\]", stripped)
        content = re.sub(r"^\[Source:[^\]]+\]\n?", "", stripped).strip()
        if content:
            chunks.append(RetrievedChunk(
                source=source_m.group(1).strip() if source_m else "unknown",
                content=content,
                score=None,
                tool=tool,
            ))
    return chunks


def _parse_tool_output(name: str, output: str) -> List[RetrievedChunk]:
    """Dispatch to the correct parser based on tool name."""
    if name in ("search_child_chunks", "fetch_parent_chunks"):
        return _parse_langchain_chunks(output, name)
    return _parse_llamaindex_chunks(output, name)


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

    final_answer = result.get("final_answer", "")

    # Graph interrupted before request_clarification — return the clarification question
    if not final_answer and not result.get("question_is_clear", True):
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

    if not final_answer:
        final_answer = "I could not generate a response."

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
            token_emitted = False
            seen_nodes: set = set()

            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                ev = event["event"]
                name = event.get("name", "")
                metadata = event.get("metadata", {})
                # True when the event belongs to the top-level LangGraph node itself
                # (not a nested chain spawned inside it).
                is_node_event = name == metadata.get("langgraph_node", "")

                # ── Progress: graph node started ──────────────────────────────
                if ev == "on_chain_start" and is_node_event and name in _NODE_PROGRESS:
                    if name not in seen_nodes:
                        seen_nodes.add(name)
                        sse = StreamChunk(type="progress", content=_NODE_PROGRESS[name])
                        yield f"data: {sse.model_dump_json()}\n\n"

                # ── Progress: rewrite_query completed — show sub-questions ────
                elif ev == "on_chain_end" and is_node_event and name == "rewrite_query":
                    output = event.get("data", {}).get("output", {})
                    questions = output.get("rewritten_questions", []) if isinstance(output, dict) else []
                    if questions:
                        q_text = " | ".join(questions)
                        sse = StreamChunk(type="progress", content=f"Sub-questions: {q_text}")
                        yield f"data: {sse.model_dump_json()}\n\n"

                # ── Progress: retrieval tool started ──────────────────────────
                elif ev == "on_tool_start" and name in _TOOL_PROGRESS:
                    sse = StreamChunk(type="progress", content=_TOOL_PROGRESS[name])
                    yield f"data: {sse.model_dump_json()}\n\n"

                # ── Retrieved chunks: retrieval tool finished ─────────────────
                elif ev == "on_tool_end" and name in _RETRIEVAL_TOOLS:
                    raw = event.get("data", {}).get("output", "")
                    # Normalize: ToolMessage objects carry content in .content
                    if hasattr(raw, "content"):
                        raw = raw.content
                    if isinstance(raw, str) and not raw.startswith(_ERROR_PREFIXES):
                        parsed = _parse_tool_output(name, raw)
                        if parsed:
                            sse = StreamChunk(
                                type="retrieved_chunk",
                                content=f"Retrieved {len(parsed)} chunk(s) via {name}",
                                retrieved_chunks=parsed,
                            )
                            yield f"data: {sse.model_dump_json()}\n\n"

                # ── Token: LLM answer streaming ───────────────────────────────
                elif ev == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        sse = StreamChunk(type="token", content=chunk.content)
                        yield f"data: {sse.model_dump_json()}\n\n"
                        token_emitted = True

            # ── Post-stream: take state snapshot ─────────────────────────────
            snapshot = await graph.aget_state(config)
            state_values = snapshot.values if snapshot else {}

            # If no tokens were streamed the graph interrupted for clarification.
            if not token_emitted:
                if not state_values.get("question_is_clear", True):
                    messages = state_values.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            sse = StreamChunk(type="token", content=msg.content)
                            yield f"data: {sse.model_dump_json()}\n\n"
                            break

            # Emit citations collected during the pipeline.
            raw_citations = state_values.get("source_citations", [])
            if raw_citations:
                citations = [Citation(**c) for c in raw_citations]
                sse = StreamChunk(type="citation", citations=citations)
                yield f"data: {sse.model_dump_json()}\n\n"

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
