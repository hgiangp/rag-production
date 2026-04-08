"""Main RAG pipeline graph — top-level LangGraph orchestration."""

from functools import partial
from typing import Optional
from urllib.parse import quote_plus

import structlog
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.graph.agent_subgraph import create_agent_subgraph
from app.graph.edges import route_after_rewrite
from app.graph.nodes.generation import aggregate_answers
from app.graph.nodes.memory import load_user_memory, save_user_memory
from app.graph.nodes.query import request_clarification, rewrite_query, summarize_history
from app.graph.state import GraphState
from app.rag.llamaindex.tools import get_llamaindex_tools
from app.services.llm import get_llm

logger = structlog.get_logger(__name__)

_graph = None
_pool: Optional[AsyncConnectionPool] = None


async def get_main_graph():
    """Singleton factory — builds the graph once and reuses it."""
    global _graph, _pool
    if _graph is not None:
        return _graph

    _graph = await _build_graph()
    return _graph


async def _run_agents(state: GraphState, agent_subgraph) -> dict:
    """Invoke the agent subgraph for each rewritten sub-question.

    LangGraph cannot automatically map GraphState.rewritten_questions →
    AgentState.question because the field names differ.  This wrapper does
    the mapping explicitly so the orchestrator always receives a non-empty
    question string.
    """
    questions = state.get("rewritten_questions") or [state.get("original_query", "")]
    all_answers: list[str] = []

    for question in questions:
        logger.info("agent_subgraph_invoked", question=question)
        result = await agent_subgraph.ainvoke({
            "messages": [],
            "question": question,
            "self_correction_count": 0,
            "user_id": state.get("user_id", ""),
        })
        answer = result.get("answer", "")
        if answer:
            all_answers.append(answer)

    return {"agent_answers": all_answers}


async def _build_graph():
    """Construct and compile the full RAG pipeline graph."""
    logger.info("building_main_graph")

    llm = get_llm(model=settings.DEFAULT_LLM_MODEL)

    # All RAG retrieval goes through LlamaIndex (heading-aware hierarchy + auto-merging)
    all_tools = get_llamaindex_tools()

    # Build agent subgraph
    agent_subgraph = create_agent_subgraph(llm=llm, tools_list=all_tools)

    # Build main graph
    builder = StateGraph(GraphState)

    builder.add_node("load_user_memory", load_user_memory)
    builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    builder.add_node("request_clarification", request_clarification)
    builder.add_node("agent", partial(_run_agents, agent_subgraph=agent_subgraph))
    builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))
    builder.add_node("save_user_memory", save_user_memory)

    builder.add_edge(START, "load_user_memory")
    builder.add_edge("load_user_memory", "summarize_history")
    builder.add_edge("summarize_history", "rewrite_query")
    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"request_clarification": "request_clarification", "spawn_agents": "agent"},
    )
    builder.add_edge("request_clarification", "rewrite_query")  # resumes after user replies
    builder.add_edge("agent", "aggregate_answers")
    builder.add_edge("aggregate_answers", "save_user_memory")
    builder.add_edge("save_user_memory", END)

    # Async PostgreSQL checkpointer for session persistence
    checkpointer = await _get_checkpointer()

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )

    logger.info("main_graph_compiled", tools_count=len(all_tools))
    return graph


async def _get_checkpointer() -> AsyncPostgresSaver:
    """Create or reuse the PostgreSQL connection pool for LangGraph checkpointing."""
    global _pool

    pw = quote_plus(settings.POSTGRES_PASSWORD)
    conn_str = (
        f"postgresql://{settings.POSTGRES_USER}:{pw}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=conn_str,
            max_size=settings.POSTGRES_POOL_SIZE,
            open=False,
            kwargs={"autocommit": True},
        )
        await _pool.open()

    checkpointer = AsyncPostgresSaver(conn=_pool)
    await checkpointer.setup()
    return checkpointer


def get_langfuse_handler(
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[CallbackHandler]:
    """Return a per-request Langfuse handler tied to a single trace.

    Pass trace_id=correlation_id so all graph nodes and LLM calls appear
    as nested spans under one trace in the Langfuse UI.
    """
    if not settings.langfuse_enabled:
        return None
    # Credentials come from the global Langfuse singleton initialised at startup.
    # CallbackHandler only accepts trace-context kwargs in Langfuse v3.
    return CallbackHandler(
        trace_id=trace_id or None,
        session_id=session_id or None,
        user_id=user_id or None,
    )
