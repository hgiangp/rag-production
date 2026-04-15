"""Chat endpoints: streaming and non-streaming RAG chat, session management, history."""

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import exists
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import CurrentUser, get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.evaluation.runner import EvalRunner
from app.graph.language import lang_name as _lang_name
from app.graph.main_graph import get_langfuse_handler, get_main_graph
from app.graph.state import GraphState
from app.models.chat_message import ChatMessage
from app.models.session import Session
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    CreateSessionRequest,
    HistoryMessage,
    Message,
    RetrievedChunk,
    SessionInfo,
    StreamChunk,
    TriadScores,
    WorkflowStep,
)
from app.services.database import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# ── Streaming helpers ──────────────────────────────────────────────────────────

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
_RETRIEVAL_TOOLS = frozenset(_TOOL_PROGRESS.keys())

# Nodes that emit workflow_step events (subset of _NODE_PROGRESS)
_WORKFLOW_NODE_PROGRESS = frozenset({"rewrite_query", "aggregate_answers"})
# Nodes NOT in _NODE_PROGRESS that still get workflow_step events
_WORKFLOW_CROSS_REF_NODES = frozenset({"detect_cross_references", "fetch_cross_ref_context"})
# Maps node name → step_type string used in WorkflowStep.step_type
_WORKFLOW_NODE_STEP_TYPES: dict = {
    "rewrite_query": "query_rewrite",
    "aggregate_answers": "aggregation",
    "detect_cross_references": "cross_ref_detected",
    "fetch_cross_ref_context": "cross_ref_fetched",
}
# Human-readable label while the node is running (status="running")
_WORKFLOW_NODE_TITLES_RUNNING: dict = {
    "rewrite_query": "Analyzing and rewriting query",
    "aggregate_answers": "Synthesizing final answer",
    "detect_cross_references": "Detecting cross-references",
    "fetch_cross_ref_context": "Fetching cross-reference context",
}

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
    chunks: List[RetrievedChunk] = []
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
    chunks: List[RetrievedChunk] = []
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
    if name in ("search_child_chunks", "fetch_parent_chunks"):
        return _parse_langchain_chunks(output, name)
    return _parse_llamaindex_chunks(output, name)


def _emit_workflow_step(step: WorkflowStep) -> str:
    """Serialize a WorkflowStep as an SSE data line."""
    sse = StreamChunk(type="workflow_step", workflow_step=step)
    return f"data: {sse.model_dump_json()}\n\n"


# ── Session title generation ───────────────────────────────────────────────────

# Names that are considered "not yet set" — LLM title generation will overwrite these.
_PLACEHOLDER_NAMES = {"", "chat", "new chat", "default", "new session"}

_TITLE_PROMPT = """\
Generate a short title for this conversation. Return ONLY the title — no quotes, \
no punctuation at the end, no explanation.

Rules:
- Maximum {max_chars} characters
- 4–7 words
- Capture the specific topic, not a generic description
- YOU MUST write the title in {target_language_name}

User: {query}
Assistant: {answer}

Title:"""


async def _generate_title_llm(query: str, answer: str, target_language: str = "en") -> str:
    """Call the LLM to produce a concise session title from the first exchange.

    The title language is determined by *target_language* (BCP-47 code), NOT by
    the content of the messages — so an English-mode session always gets an
    English title even when the retrieved documents contain Japanese text.

    Returns an empty string on any failure so the caller can fall back gracefully.
    """
    from app.services.llm import get_llm

    max_chars = settings.SESSION_TITLE_MAX_CHARS
    # Truncate inputs so the title prompt stays cheap (≈ 300 tokens total).
    prompt = _TITLE_PROMPT.format(
        max_chars=max_chars,
        target_language_name=_lang_name(target_language),
        query=query[:300],
        answer=answer[:300],
    )
    try:
        llm = get_llm(model=settings.DEFAULT_LLM_MODEL)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        title = response.content.strip().strip('"').strip("'")
        # Hard-cap at max_chars in case the model ignores the instruction
        if len(title) > max_chars:
            title = title[:max_chars - 1].rsplit(" ", 1)[0] + "…"
        return title
    except Exception as exc:
        logger.warning("session_title_generation_failed", error=str(exc))
        return ""


async def _set_session_title(session_id: UUID, query: str, answer: str, target_language: str = "en") -> None:
    """Fire-and-forget coroutine: generates an LLM title and writes it to the DB.

    Falls back to a truncated version of the query if the LLM call fails.
    Uses its own DB session so it can run independently of the request lifecycle.
    """
    title = await _generate_title_llm(query, answer, target_language)
    if not title:
        # Fallback: truncate query at word boundary
        text = " ".join(query.split())
        max_chars = settings.SESSION_TITLE_MAX_CHARS
        title = text if len(text) <= max_chars else text[:max_chars - 1].rsplit(" ", 1)[0] + "…"

    from app.services.database import _async_session_factory
    async with _async_session_factory() as db:
        result = await db.exec(select(Session).where(Session.id == session_id))
        sess = result.first()
        if sess:
            sess.name = title
            db.add(sess)
            await db.commit()
    logger.info("session_title_set", session_id=str(session_id), title=title)


# ── Message persistence ────────────────────────────────────────────────────────

async def _save_turn(
    db: AsyncSession,
    user_id: UUID,
    session_id: UUID,
    query: str,
    answer: str,
    citations: List[Citation],
    eval_scores: Optional[TriadScores] = None,
    target_language: str = "en",
    workflow_steps: Optional[List[dict]] = None,
) -> None:
    """Persist one user + assistant turn to chat_message.

    On the very first turn fires a background task that asks the LLM to produce
    a short title from the exchange and writes it to session.name.
    The title is written in *target_language* so the session name always matches
    the UI language selection, regardless of document language.
    session.updated_at is bumped by the DB trigger on INSERT.
    """
    # Check before inserting so we know whether this is the first turn.
    count_result = await db.exec(
        select(func.count(ChatMessage.id)).where(ChatMessage.session_id == session_id)
    )
    is_first_turn = count_result.one() == 0

    user_msg = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=query,
        citations=[],
    )
    scores_dict = eval_scores.model_dump() if eval_scores else None
    asst_msg = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=answer,
        citations=[c.model_dump() for c in citations],
        eval_scores=scores_dict,
        workflow_steps=workflow_steps or [],
    )
    db.add(user_msg)
    db.add(asst_msg)
    await db.commit()

    # Fire title generation in the background — same pattern as eval (Rule R7).
    if is_first_turn:
        sess_result = await db.exec(select(Session).where(Session.id == session_id))
        sess = sess_result.first()
        if sess and sess.name.strip().lower() in _PLACEHOLDER_NAMES:
            asyncio.create_task(_set_session_title(session_id, query, answer, target_language))

    logger.info("chat_turn_saved", session_id=str(session_id), answer_length=len(answer), first_turn=is_first_turn)


def _build_initial_state(body: ChatRequest, current_user: CurrentUser, correlation_id: str, session_id: UUID) -> GraphState:
    return GraphState(
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
        target_language=body.target_language or settings.DEFAULT_TARGET_LANGUAGE,
        query_language="",
    )


# ── Session endpoints ──────────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionInfo])
@limiter.limit("60/minute")
async def list_sessions(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> List[SessionInfo]:
    """List chat sessions for the current user that have at least one message, newest first.

    Empty sessions (created at login but never used) are excluded.
    """
    result = await db.exec(
        select(Session)
        .where(
            Session.user_id == current_user.user_id,
            exists().where(ChatMessage.session_id == Session.id),
        )
        .order_by(Session.updated_at.desc())
        .limit(50)
    )
    sessions = result.all()

    infos: List[SessionInfo] = []
    for sess in sessions:
        # Message count
        count_result = await db.exec(
            select(func.count(ChatMessage.id)).where(ChatMessage.session_id == sess.id)
        )
        msg_count = count_result.one()

        # Last assistant message preview
        last_result = await db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == sess.id, ChatMessage.role == "assistant")
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
        last_msg = last_result.first()
        preview = (last_msg.content[:80] + "…") if last_msg and len(last_msg.content) > 80 else (last_msg.content if last_msg else None)

        infos.append(SessionInfo(
            session_id=sess.id,
            name=sess.name or "Chat",
            created_at=sess.created_at,
            updated_at=sess.updated_at,
            message_count=msg_count,
            last_message=preview,
        ))
    return infos


@router.post("/sessions", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
async def create_session(
    request: Request,
    body: CreateSessionRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> SessionInfo:
    """Create a new chat session. The title is set automatically from the first message."""
    sess = Session(user_id=current_user.user_id, name="")
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    logger.info("session_created", session_id=str(sess.id), user_id=str(current_user.user_id))
    return SessionInfo(
        session_id=sess.id,
        name=sess.name,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
        message_count=0,
        last_message=None,
    )


@router.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
@limiter.limit("60/minute")
async def get_session_history(
    request: Request,
    session_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ChatHistoryResponse:
    """Return the full message history for a session owned by the current user."""
    sess_result = await db.exec(
        select(Session).where(Session.id == session_id, Session.user_id == current_user.user_id)
    )
    sess = sess_result.first()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    msgs_result = await db.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = msgs_result.all()

    history: List[HistoryMessage] = []
    for m in messages:
        citations = [Citation(**c) for c in (m.citations or [])]
        scores = TriadScores(**m.eval_scores) if m.eval_scores else None
        steps: List[WorkflowStep] = []
        for s in (m.workflow_steps or []):
            try:
                steps.append(WorkflowStep.model_validate(s))
            except Exception:
                pass
        history.append(HistoryMessage(
            id=m.id,
            role=m.role,
            content=m.content,
            citations=citations,
            eval_scores=scores,
            workflow_steps=steps,
            created_at=m.created_at,
        ))

    return ChatHistoryResponse(session_id=session_id, name=sess.name or "Chat", messages=history)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
async def delete_session(
    request: Request,
    session_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete a session and all its messages (cascades via FK)."""
    sess_result = await db.exec(
        select(Session).where(Session.id == session_id, Session.user_id == current_user.user_id)
    )
    sess = sess_result.first()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    await db.delete(sess)
    await db.commit()
    logger.info("session_deleted", session_id=str(session_id), user_id=str(current_user.user_id))


# ── Chat endpoints ─────────────────────────────────────────────────────────────

@router.post("/invoke", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat_invoke(
    request: Request,
    body: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ChatResponse:
    """Non-streaming chat: returns full response after pipeline completes."""
    start_ms = int(time.time() * 1000)
    correlation_id = getattr(request.state, "correlation_id", "")
    session_id = body.session_id or current_user.session_id

    graph = await get_main_graph()
    initial_state = _build_initial_state(body, current_user, correlation_id, session_id)

    langfuse_handler = get_langfuse_handler(trace_id=correlation_id)
    config = {
        "configurable": {"thread_id": f"{current_user.user_id}:{session_id}"},
        "run_name": "rag_chat",
        "callbacks": [langfuse_handler] if langfuse_handler else [],
    }

    try:
        result: GraphState = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("graph_invocation_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pipeline failed")

    final_answer = result.get("final_answer", "")

    if not final_answer and not result.get("question_is_clear", True):
        for msg in reversed(result.get("messages", [])):
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

    # Persist the turn
    await _save_turn(
        db=db,
        user_id=current_user.user_id,
        session_id=UUID(str(session_id)),
        query=body.message,
        answer=final_answer,
        citations=citations,
        target_language=body.target_language or settings.DEFAULT_TARGET_LANGUAGE,
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
        initial_state = _build_initial_state(body, current_user, correlation_id, session_id)
        langfuse_handler = get_langfuse_handler(trace_id=correlation_id)
        config = {
            "configurable": {"thread_id": f"{current_user.user_id}:{session_id}"},
            "run_name": "rag_chat_stream",
            "callbacks": [langfuse_handler] if langfuse_handler else [],
        }

        try:
            token_emitted = False
            seen_nodes: set = set()
            streamed_tokens: List[str] = []
            # Per-run timing: keyed by event run_id (unique per node/tool invocation)
            step_start_times: dict = {}
            # Filled by rewrite_query, read by aggregate_answers to count sub-answers
            collected_sub_questions: List[str] = []
            # Completed workflow steps to persist to DB
            workflow_steps_log: List[dict] = []

            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                ev = event["event"]
                name = event.get("name", "")
                run_id = event.get("run_id") or name
                metadata = event.get("metadata", {})
                is_node_event = name == metadata.get("langgraph_node", "")

                # ── on_chain_start for nodes in _NODE_PROGRESS ─────────────────
                if ev == "on_chain_start" and is_node_event and name in _NODE_PROGRESS:
                    if name not in seen_nodes:
                        seen_nodes.add(name)
                        sse = StreamChunk(type="progress", content=_NODE_PROGRESS[name])
                        yield f"data: {sse.model_dump_json()}\n\n"
                    if name in _WORKFLOW_NODE_PROGRESS:
                        step_start_times[run_id] = int(time.time() * 1000)
                        yield _emit_workflow_step(WorkflowStep(
                            step_type=_WORKFLOW_NODE_STEP_TYPES[name],
                            title=_WORKFLOW_NODE_TITLES_RUNNING[name],
                            status="running",
                        ))

                # ── on_chain_end for rewrite_query ─────────────────────────────
                elif ev == "on_chain_end" and is_node_event and name == "rewrite_query":
                    output = event.get("data", {}).get("output", {})
                    questions = output.get("rewritten_questions", []) if isinstance(output, dict) else []
                    lang = output.get("query_language", "en") if isinstance(output, dict) else "en"
                    if questions:
                        sse = StreamChunk(type="progress", content=f"Sub-questions: {' | '.join(questions)}")
                        yield f"data: {sse.model_dump_json()}\n\n"
                    collected_sub_questions = questions
                    duration = int(time.time() * 1000) - step_start_times.pop(run_id, int(time.time() * 1000))
                    step = WorkflowStep(
                        step_type="query_rewrite",
                        title=f"Query rewritten — {len(questions)} sub-question(s)",
                        status="done",
                        data={"sub_questions": questions, "detected_language": lang},
                        duration_ms=duration,
                    )
                    workflow_steps_log.append(step.model_dump())
                    yield _emit_workflow_step(step)

                # ── on_chain_start for cross-ref nodes (not in _NODE_PROGRESS) ──
                elif ev == "on_chain_start" and is_node_event and name in _WORKFLOW_CROSS_REF_NODES:
                    step_start_times[run_id] = int(time.time() * 1000)
                    yield _emit_workflow_step(WorkflowStep(
                        step_type=_WORKFLOW_NODE_STEP_TYPES[name],
                        title=_WORKFLOW_NODE_TITLES_RUNNING[name],
                        status="running",
                    ))

                # ── on_chain_end for detect_cross_references ───────────────────
                elif ev == "on_chain_end" and is_node_event and name == "detect_cross_references":
                    output = event.get("data", {}).get("output", {})
                    targets = output.get("cross_ref_targets", []) if isinstance(output, dict) else []
                    labels: List[str] = []
                    for t in (targets or []):
                        if isinstance(t, str):
                            labels.append(t)
                        elif isinstance(t, dict):
                            spec = t.get("spec_name") or t.get("spec_id") or "?"
                            section = t.get("section_number") or t.get("section") or ""
                            labels.append(f"{spec} § {section}" if section else spec)
                    duration = int(time.time() * 1000) - step_start_times.pop(run_id, int(time.time() * 1000))
                    step = WorkflowStep(
                        step_type="cross_ref_detected",
                        title=f"Found {len(labels)} cross-reference(s)",
                        status="done",
                        data={"references": labels, "count": len(labels)},
                        duration_ms=duration,
                    )
                    workflow_steps_log.append(step.model_dump())
                    yield _emit_workflow_step(step)

                # ── on_chain_end for fetch_cross_ref_context ───────────────────
                elif ev == "on_chain_end" and is_node_event and name == "fetch_cross_ref_context":
                    output = event.get("data", {}).get("output", {})
                    injected = 1 if (isinstance(output, dict) and output.get("messages")) else 0
                    duration = int(time.time() * 1000) - step_start_times.pop(run_id, int(time.time() * 1000))
                    step = WorkflowStep(
                        step_type="cross_ref_fetched",
                        title="Cross-reference context fetched",
                        status="done",
                        data={"contexts_injected": injected},
                        duration_ms=duration,
                    )
                    workflow_steps_log.append(step.model_dump())
                    yield _emit_workflow_step(step)

                # ── on_chain_end for aggregate_answers ─────────────────────────
                elif ev == "on_chain_end" and is_node_event and name == "aggregate_answers":
                    n = len(collected_sub_questions) or 1
                    duration = int(time.time() * 1000) - step_start_times.pop(run_id, int(time.time() * 1000))
                    step = WorkflowStep(
                        step_type="aggregation",
                        title=f"Synthesized {n} sub-answer(s)",
                        status="done",
                        data={"sub_answer_count": n},
                        duration_ms=duration,
                    )
                    workflow_steps_log.append(step.model_dump())
                    yield _emit_workflow_step(step)

                # ── on_tool_start ──────────────────────────────────────────────
                elif ev == "on_tool_start" and name in _TOOL_PROGRESS:
                    sse = StreamChunk(type="progress", content=_TOOL_PROGRESS[name])
                    yield f"data: {sse.model_dump_json()}\n\n"
                    if name in _RETRIEVAL_TOOLS:
                        step_start_times[run_id] = int(time.time() * 1000)
                        yield _emit_workflow_step(WorkflowStep(
                            step_type="tool_retrieval",
                            title=_TOOL_PROGRESS[name].rstrip("."),
                            status="running",
                        ))

                # ── on_tool_end ────────────────────────────────────────────────
                elif ev == "on_tool_end" and name in _RETRIEVAL_TOOLS:
                    raw = event.get("data", {}).get("output", "")
                    if hasattr(raw, "content"):
                        raw = raw.content
                    duration = int(time.time() * 1000) - step_start_times.pop(run_id, int(time.time() * 1000))
                    if isinstance(raw, str) and not raw.startswith(_ERROR_PREFIXES):
                        parsed = _parse_tool_output(name, raw)
                        if parsed:
                            sse = StreamChunk(
                                type="retrieved_chunk",
                                content=f"Retrieved {len(parsed)} chunk(s) via {name}",
                                retrieved_chunks=parsed,
                            )
                            yield f"data: {sse.model_dump_json()}\n\n"
                            step = WorkflowStep(
                                step_type="tool_retrieval",
                                title=f"Retrieved {len(parsed)} chunk(s) via {name}",
                                status="done",
                                data={
                                    "tool_name": name,
                                    "chunks_found": len(parsed),
                                    "chunks": [c.model_dump() for c in parsed],
                                },
                                duration_ms=duration,
                            )
                        else:
                            step = WorkflowStep(
                                step_type="tool_retrieval",
                                title=f"No results via {name}",
                                status="done",
                                data={"tool_name": name, "chunks_found": 0, "chunks": []},
                                duration_ms=duration,
                            )
                    else:
                        step = WorkflowStep(
                            step_type="tool_retrieval",
                            title=f"No results via {name}",
                            status="done",
                            data={"tool_name": name, "chunks_found": 0, "chunks": []},
                            duration_ms=duration,
                        )
                    workflow_steps_log.append(step.model_dump())
                    yield _emit_workflow_step(step)

                # ── LLM token stream ───────────────────────────────────────────
                elif ev == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        streamed_tokens.append(chunk.content)
                        sse = StreamChunk(type="token", content=chunk.content)
                        yield f"data: {sse.model_dump_json()}\n\n"
                        token_emitted = True

            # State snapshot after streaming completes
            snapshot = await graph.aget_state(config)
            state_values = snapshot.values if snapshot else {}

            if not token_emitted:
                if not state_values.get("question_is_clear", True):
                    for msg in reversed(state_values.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content:
                            streamed_tokens.append(msg.content)
                            sse = StreamChunk(type="token", content=msg.content)
                            yield f"data: {sse.model_dump_json()}\n\n"
                            break

            raw_citations = state_values.get("source_citations", [])
            citations = [Citation(**c) for c in raw_citations] if raw_citations else []
            if citations:
                sse = StreamChunk(type="citation", citations=citations)
                yield f"data: {sse.model_dump_json()}\n\n"

            # Persist the turn from separate DB session (generator context)
            final_answer = state_values.get("final_answer", "") or "".join(streamed_tokens)
            if final_answer:
                from app.services.database import _async_session_factory
                async with _async_session_factory() as db:
                    await _save_turn(
                        db=db,
                        user_id=current_user.user_id,
                        session_id=UUID(str(session_id)),
                        query=body.message,
                        answer=final_answer,
                        citations=citations,
                        target_language=body.target_language or settings.DEFAULT_TARGET_LANGUAGE,
                        workflow_steps=workflow_steps_log,
                    )

            yield f"data: {StreamChunk(type='done').model_dump_json()}\n\n"

        except Exception as exc:
            logger.exception("stream_error", error=str(exc))
            yield f"data: {StreamChunk(type='error', error=str(exc)).model_dump_json()}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
