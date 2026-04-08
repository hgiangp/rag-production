"""Agent subgraph — handles a single sub-question with tool-calling loop."""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.graph.edges import (
    route_after_compression_check,
    route_after_cross_ref_detection,
    route_after_orchestrator,
)
from app.graph.nodes.generation import (
    collect_answer,
    compress_context,
    fallback_response,
    orchestrator,
)
from app.graph.nodes.retrieval import (
    detect_cross_references,
    fetch_cross_ref_context,
    should_compress_context,
)
from app.graph.state import AgentState


def create_agent_subgraph(llm: BaseChatModel, tools_list: list):
    """Build and compile the agent reasoning subgraph.

    Args:
        llm: Base LLM (will have tools bound internally)
        tools_list: All LangGraph tools available to the agent

    Returns:
        Compiled agent subgraph
    """
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    builder.add_node("tools", tool_node)
    builder.add_node("detect_cross_references", partial(detect_cross_references, llm=llm))
    builder.add_node("fetch_cross_ref_context", fetch_cross_ref_context)
    builder.add_node("should_compress_context", should_compress_context)
    builder.add_node("compress_context", partial(compress_context, llm=llm))
    builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    builder.add_node("collect_answer", collect_answer)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "tools": "tools",
            "collect_answer": "collect_answer",
            "fallback_response": "fallback_response",
        },
    )
    # After tool calls: detect cross-references before compression check.
    builder.add_edge("tools", "detect_cross_references")
    builder.add_conditional_edges(
        "detect_cross_references",
        route_after_cross_ref_detection,
        {
            "fetch_cross_ref_context": "fetch_cross_ref_context",
            "should_compress_context": "should_compress_context",
        },
    )
    builder.add_edge("fetch_cross_ref_context", "should_compress_context")
    builder.add_conditional_edges(
        "should_compress_context",
        route_after_compression_check,
        {"compress_context": "compress_context", "orchestrator": "orchestrator"},
    )
    builder.add_edge("compress_context", "orchestrator")
    builder.add_edge("fallback_response", "collect_answer")
    builder.add_edge("collect_answer", END)

    return builder.compile()
