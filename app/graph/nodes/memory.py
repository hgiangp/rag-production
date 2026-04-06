"""Memory nodes: load and save per-user long-term memory via mem0ai."""

import structlog
from langchain_core.messages import SystemMessage

from app.core.metrics import GRAPH_NODE_COUNT, GRAPH_NODE_LATENCY
from app.graph.state import GraphState
from app.memory.longterm import LongTermMemory

logger = structlog.get_logger(__name__)
_ltm = LongTermMemory()


async def load_user_memory(state: GraphState) -> dict:
    """Retrieve relevant long-term memories and prepend as SystemMessage."""
    import time
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="load_user_memory").inc()

    user_id = state.get("user_id", "")
    query = state.get("original_query", "")

    memories = await _ltm.search(user_id=user_id, query=query)
    if not memories:
        return {}

    memory_text = "\n".join(f"- {m}" for m in memories)
    system_msg = SystemMessage(content=f"[User context from previous conversations]\n{memory_text}")

    logger.info("user_memory_loaded", user_id=user_id, memory_count=len(memories))
    GRAPH_NODE_LATENCY.labels(node="load_user_memory").observe(time.perf_counter() - start)

    return {"messages": [system_msg]}


async def save_user_memory(state: GraphState) -> dict:
    """Extract and persist new facts from the completed conversation turn."""
    import time
    start = time.perf_counter()
    GRAPH_NODE_COUNT.labels(node="save_user_memory").inc()

    user_id = state.get("user_id", "")
    query = state.get("original_query", "")
    answer = state.get("final_answer", "")

    if query and answer:
        await _ltm.add(
            user_id=user_id,
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ],
        )
        logger.info("user_memory_saved", user_id=user_id)

    GRAPH_NODE_LATENCY.labels(node="save_user_memory").observe(time.perf_counter() - start)
    return {}
