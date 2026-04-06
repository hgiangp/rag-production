"""LangGraph tools wrapping the LangChain/Qdrant retrieval layer.

Tools exposed to the agent:
- search_child_chunks: ANN search on child chunks
- fetch_parent_chunks: fetch full parent context by ID list
"""

import time
from typing import Annotated, List

import structlog
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langgraph.prebuilt import InjectedState

from app.core.config import settings
from app.core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_RESULTS
from app.graph.state import AgentState
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store_service

logger = structlog.get_logger(__name__)


def get_langchain_tools() -> List:
    """Build and return all LangChain RAG tools. Collection is resolved per-request from agent state."""

    @tool("search_child_chunks")
    async def search_child_chunks(
        query: str,
        state: Annotated[AgentState, InjectedState],
        limit: int = settings.MAX_RETRIEVAL_K,
    ) -> str:
        """Search for relevant document passages. Use this FIRST for any factual question.

        Args:
            query: The search query
            limit: Maximum number of results (default 5)

        Returns:
            Formatted string with matched passages and their parent IDs
        """
        collection = f"docs_{state['user_id']}"
        child_collection = f"{collection}__{settings.QDRANT_CHILD_COLLECTION}"
        start = time.perf_counter()
        try:
            store = QdrantVectorStore.from_existing_collection(
                embedding=embedding_service.model,
                url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                collection_name=child_collection,
                prefer_grpc=settings.QDRANT_PREFER_GRPC,
            )
            results = await store.asimilarity_search_with_relevance_scores(
                query=query, k=limit, score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD
            )
            RETRIEVAL_LATENCY.labels(tool="search_child_chunks", collection=collection).observe(
                time.perf_counter() - start
            )
            RETRIEVAL_RESULTS.labels(tool="search_child_chunks").observe(len(results))

            if not results:
                logger.info("retrieval_no_results", query=query)
                return "NO_RELEVANT_CHUNKS: No relevant passages found. Try different keywords."

            logger.info("retrieval_success", query=query, results=len(results))
            return "\n\n".join(
                f"Parent ID: {doc.metadata.get('parent_id', 'unknown')}\n"
                f"Source: {doc.metadata.get('filename', 'unknown')}\n"
                f"Relevance: {score:.2f}\n"
                f"Content: {doc.page_content.strip()}"
                for doc, score in results
            )
        except Exception as exc:
            logger.exception("search_child_chunks_failed", error=str(exc))
            return f"RETRIEVAL_ERROR: {str(exc)}"

    @tool("fetch_parent_chunks")
    async def fetch_parent_chunks(
        parent_ids: List[str],
        state: Annotated[AgentState, InjectedState],
    ) -> str:
        """Retrieve full context for document sections by their parent IDs.

        Call this AFTER search_child_chunks to get the complete context for matched passages.

        Args:
            parent_ids: List of parent chunk IDs from search_child_chunks results

        Returns:
            Full content of each parent chunk
        """
        collection = f"docs_{state['user_id']}"
        parent_collection = f"{collection}__{settings.QDRANT_PARENT_COLLECTION}"
        start = time.perf_counter()
        try:
            client = vector_store_service.client

            results = []
            for pid in parent_ids[:10]:  # cap at 10 to avoid token explosion
                hits = client.scroll(
                    collection_name=parent_collection,
                    scroll_filter={"must": [{"key": "parent_id", "match": {"value": pid}}]},
                    limit=1,
                )
                if hits[0]:
                    payload = hits[0][0].payload
                    results.append(
                        f"Parent ID: {pid}\n"
                        f"Source: {payload.get('filename', 'unknown')}\n"
                        f"Content: {payload.get('content', '').strip()}"
                    )

            RETRIEVAL_LATENCY.labels(tool="fetch_parent_chunks", collection=collection).observe(
                time.perf_counter() - start
            )

            if not results:
                return "NO_PARENT_DOCUMENTS: Could not retrieve parent context."

            logger.info("parent_chunks_fetched", count=len(results))
            return "\n\n---\n\n".join(results)
        except Exception as exc:
            logger.exception("fetch_parent_chunks_failed", error=str(exc))
            return f"PARENT_RETRIEVAL_ERROR: {str(exc)}"

    return [search_child_chunks, fetch_parent_chunks]
