"""Long-term memory via mem0ai — per-user cross-session memory."""

from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LongTermMemory:
    """Wraps mem0ai AsyncMemory for per-user persistent memory."""

    def __init__(self) -> None:
        self._memory = None
        self._collection_ready = False

    async def _get_embedding_dim(self) -> int:
        """Probe the configured embedder to get vector dimension."""
        provider = settings.LLM_PROVIDER
        if provider == "ollama":
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": settings.MEM0_EMBEDDER_MODEL, "prompt": "dim_probe"},
                    timeout=30.0,
                )
                resp.raise_for_status()
                return len(resp.json()["embedding"])
        else:  # openai or anthropic (both use openai embedder in mem0)
            from openai import AsyncOpenAI
            api_key = settings.OPENAI_API_KEY
            client = AsyncOpenAI(api_key=api_key)
            result = await client.embeddings.create(
                input="dim_probe", model=settings.MEM0_EMBEDDER_MODEL
            )
            return len(result.data[0].embedding)

    async def _ensure_collection(self) -> None:
        """Create (or recreate on dim mismatch) the Qdrant LTM collection."""
        if self._collection_ready:
            return
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        try:
            dim = await self._get_embedding_dim()
            collections = await client.get_collections()
            existing = {c.name for c in collections.collections}
            collection = settings.LONG_TERM_MEMORY_COLLECTION
            if collection in existing:
                info = await client.get_collection(collection)
                current_dim = info.config.params.vectors.size
                if current_dim != dim:
                    logger.warning(
                        "ltm_collection_dim_mismatch",
                        collection=collection,
                        current_dim=current_dim,
                        expected_dim=dim,
                        action="recreating",
                    )
                    await client.delete_collection(collection)
                    existing.discard(collection)
            if collection not in existing:
                await client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info("ltm_collection_created", collection=collection, dim=dim)
            self._collection_ready = True
        finally:
            await client.close()

    def _get_memory(self):
        if self._memory is None:
            from mem0 import AsyncMemory
            provider = settings.LLM_PROVIDER
            if provider == "ollama":
                llm_cfg = {
                    "provider": "ollama",
                    "config": {"model": settings.MEM0_LLM_MODEL, "ollama_base_url": settings.OLLAMA_BASE_URL},
                }
                embedder_cfg = {
                    "provider": "ollama",
                    "config": {"model": settings.MEM0_EMBEDDER_MODEL, "ollama_base_url": settings.OLLAMA_BASE_URL},
                }
            elif provider == "anthropic":
                llm_cfg = {
                    "provider": "anthropic",
                    "config": {"model": settings.MEM0_LLM_MODEL, "api_key": settings.ANTHROPIC_API_KEY},
                }
                embedder_cfg = {
                    "provider": "openai",
                    "config": {"model": settings.MEM0_EMBEDDER_MODEL, "api_key": settings.OPENAI_API_KEY},
                }
            else:  # openai
                llm_cfg = {
                    "provider": "openai",
                    "config": {"model": settings.MEM0_LLM_MODEL, "api_key": settings.OPENAI_API_KEY},
                }
                embedder_cfg = {
                    "provider": "openai",
                    "config": {"model": settings.MEM0_EMBEDDER_MODEL, "api_key": settings.OPENAI_API_KEY},
                }
            config = {
                "llm": llm_cfg,
                "embedder": embedder_cfg,
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": settings.QDRANT_HOST,
                        "port": settings.QDRANT_PORT,
                        "collection_name": settings.LONG_TERM_MEMORY_COLLECTION,
                    },
                },
            }
            self._memory = AsyncMemory.from_config(config)
        return self._memory

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def add(self, user_id: str, messages: List[Dict[str, str]]) -> None:
        """
        Add conversation turn to long-term memory.

        Args:
            user_id: The user identifier.
            messages: List of message dicts, each with required keys:
                - "role": str, e.g., "user" or "assistant"
                - "content": str, the message text
        """
        try:
            await self._ensure_collection()
            memory = self._get_memory()
            await memory.add(messages, user_id=user_id)
            logger.info("ltm_memory_added", user_id=user_id)
        except Exception as exc:
            logger.warning("ltm_add_failed", user_id=user_id, error=str(exc))

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def search(self, user_id: str, query: str, limit: int = 5) -> List[str]:
        """Search for relevant memories for a given query."""
        try:
            await self._ensure_collection()
            memory = self._get_memory()
            results = await memory.search(query=query, user_id=user_id, limit=limit)
            return [r["memory"] for r in results.get("results", [])]
        except Exception as exc:
            logger.warning("ltm_search_failed", user_id=user_id, error=str(exc))
            return []

    async def delete_all(self, user_id: str) -> None:
        """Delete all memories for a user."""
        try:
            await self._ensure_collection()
            memory = self._get_memory()
            await memory.delete_all(user_id=user_id)
            logger.info("ltm_memories_deleted", user_id=user_id)
        except Exception as exc:
            logger.warning("ltm_delete_failed", user_id=user_id, error=str(exc))
