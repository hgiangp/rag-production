"""Agent subgraph — handles a single sub-question with tool-calling loop."""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.graph.edges import route_after_compression_check, route_after_orchestrator
from app.graph.nodes.generation import (
    collect_answer,
    compress_context,
    fallback_response,
    orchestrator,
)
from app.graph.nodes.retrieval import should_compress_context
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
    builder.add_edge("tools", "should_compress_context")
    builder.add_conditional_edges(
        "should_compress_context",
        route_after_compression_check,
        {"compress_context": "compress_context", "orchestrator": "orchestrator"},
    )
    builder.add_edge("compress_context", "orchestrator")
    builder.add_edge("fallback_response", "collect_answer")
    builder.add_edge("collect_answer", END)

    return builder.compile()
