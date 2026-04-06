"""Generation nodes: orchestrator, aggregate answers, fallback, collect answer."""

import time

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import settings
from app.core.metrics import GRAPH_NODE_COUNT, GRAPH_NODE_LATENCY, LLM_LATENCY
from app.graph.state import AgentState, GraphState

logger = structlog.get_logger(__name__)

_ORCHESTRATOR_SYSTEM = (
    "You are a research assistant with access to document search tools. "
    "To answer the user's question:\n"
    "1. ALWAYS start by calling search_child_chunks to find relevant passages\n"
    "2. Then call fetch_parent_chunks to get full context\n"
    "3. If the retrieved context is insufficient, search again with different keywords\n"
    "4. Once you have enough context, provide a comprehensive answer with source references\n"
    "Never answer from memory alone — always ground answers in retrieved documents."
)

_FALLBACK_SYSTEM = (
    "The document search did not return sufficient results for the user's question. "
    "Acknowledge this honestly and suggest what information the user might need to provide "
    "or what alternative approach might help. Do not fabricate information."
)

_AGGREGATE_SYSTEM = (
    "You are synthesizing answers from multiple research agents. "
    "Combine their findings into a single, coherent, well-structured answer. "
    "Eliminate redundancy, resolve any contradictions, and cite sources appropriately. "
    "Maintain factual accuracy — only include information from the provided answers."
)


async def orchestrator(state: AgentState, llm_with_tools: BaseChatModel) -> dict:
    """Main agent reasoning node — calls LLM with tool bindings."""
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="orchestrator").inc()

    sys_msg = SystemMessage(content=_ORCHESTRATOR_SYSTEM)
    context_summary = state.get("context_summary", "").strip()

    injection = []
    if context_summary:
        injection = [HumanMessage(content=f"[Prior research summary]\n{context_summary}")]

    if not state.get("messages"):
        question = state.get("question", "")
        force = HumanMessage(content="Start by calling search_child_chunks to find relevant information.")
        msgs = [sys_msg] + injection + [HumanMessage(content=question), force]
    else:
        msgs = [sys_msg] + injection + list(state["messages"])

    response = await llm_with_tools.ainvoke(msgs)

    LLM_LATENCY.labels(model=settings.DEFAULT_LLM_MODEL, node="orchestrator").observe(time.perf_counter() - start)
    GRAPH_NODE_LATENCY.labels(node="orchestrator").observe(time.perf_counter() - start)
    return {"messages": [response]}


async def compress_context(state: AgentState, llm: BaseChatModel) -> dict:
    """Summarize tool call results to stay within token budget."""
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="compress_context").inc()

    from langchain_core.messages import ToolMessage
    tool_results = [
        m.content for m in state.get("messages", [])
        if isinstance(m, ToolMessage) and m.content
    ]
    if not tool_results:
        return {}

    combined = "\n\n".join(tool_results[-6:])  # last 6 tool results
    prompt = (
        f"Summarize the following research findings concisely, "
        f"preserving key facts and source references:\n\n{combined}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])

    logger.info("context_compressed", original_length=len(combined), summary_length=len(resp.content))
    GRAPH_NODE_LATENCY.labels(node="compress_context").observe(time.perf_counter() - start)
    return {"context_summary": resp.content, "self_correction_count": state.get("self_correction_count", 0) + 1}


async def fallback_response(state: AgentState, llm: BaseChatModel) -> dict:
    """Generate an honest fallback when retrieval is insufficient."""
    GRAPH_NODE_COUNT.labels(node="fallback_response").inc()

    question = state.get("question", "the question")
    resp = await llm.ainvoke([
        SystemMessage(content=_FALLBACK_SYSTEM),
        HumanMessage(content=f"Question: {question}"),
    ])
    logger.info("fallback_response_generated")
    return {"messages": [AIMessage(content=resp.content)], "answer": resp.content}


def collect_answer(state: AgentState) -> dict:
    """Extract the last AIMessage as the agent's final answer."""
    GRAPH_NODE_COUNT.labels(node="collect_answer").inc()

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return {"answer": msg.content}
    return {"answer": ""}


async def aggregate_answers(state: GraphState, llm: BaseChatModel) -> dict:
    """Merge answers from all agent subgraphs into a single final response."""
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="aggregate_answers").inc()

    answers = state.get("agent_answers", [])
    if not answers:
        return {"final_answer": "I could not find relevant information to answer your question."}

    if len(answers) == 1:
        GRAPH_NODE_LATENCY.labels(node="aggregate_answers").observe(time.perf_counter() - start)
        return {"final_answer": answers[0]}

    combined = "\n\n---\n\n".join(f"[Answer {i+1}]\n{a}" for i, a in enumerate(answers))
    prompt = f"Original question: {state.get('original_query', '')}\n\nAnswers to synthesize:\n{combined}"

    resp = await llm.ainvoke([
        SystemMessage(content=_AGGREGATE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    LLM_LATENCY.labels(model=settings.DEFAULT_LLM_MODEL, node="aggregate_answers").observe(time.perf_counter() - start)
    GRAPH_NODE_LATENCY.labels(node="aggregate_answers").observe(time.perf_counter() - start)
    return {"final_answer": resp.content}
