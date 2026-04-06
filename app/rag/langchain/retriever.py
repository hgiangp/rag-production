"""HierarchicalRetriever: child similarity search → parent chunk fetch."""

from typing import List

import structlog

from app.core.config import settings
from app.services.vector_store import vector_store_service

logger = structlog.get_logger(__name__)


class HierarchicalRetriever:
    """Two-stage retriever: ANN on child chunks, full context from parent chunks."""

    def __init__(self, collection: str) -> None:
        self.collection = collection
        self._child_collection = f"{collection}__{settings.QDRANT_CHILD_COLLECTION}"
        self._parent_collection = f"{collection}__{settings.QDRANT_PARENT_COLLECTION}"

    async def retrieve(self, query: str, k: int = settings.MAX_RETRIEVAL_K) -> List[dict]:
        """Search child chunks and return their parent contexts."""
        from langchain_qdrant import QdrantVectorStore
        from app.services.embedding import embedding_service

        store = QdrantVectorStore.from_existing_collection(
            embedding=embedding_service.model,
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            collection_name=self._child_collection,
        )
        hits = await store.asimilarity_search_with_relevance_scores(
            query=query, k=k, score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD
        )

        parent_ids = list({doc.metadata.get("parent_id") for doc, _ in hits if doc.metadata.get("parent_id")})
        parents = []
        client = vector_store_service.client

        for pid in parent_ids[:k]:
            results = client.scroll(
                collection_name=self._parent_collection,
                scroll_filter={"must": [{"key": "parent_id", "match": {"value": pid}}]},
                limit=1,
            )
            if results[0]:
                p = results[0][0].payload
                parents.append({
                    "parent_id": pid,
                    "content": p.get("content", ""),
                    "filename": p.get("filename", "unknown"),
                    "document_id": p.get("document_id", ""),
                })

        logger.info("hierarchical_retrieval_done", query=query, parents_fetched=len(parents))
        return parents
