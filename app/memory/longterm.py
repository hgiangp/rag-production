"""Long-term memory via mem0ai — per-user cross-session memory."""

from typing import Any, Dict, List, Optional

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LongTermMemory:
    """Wraps mem0ai AsyncMemory for per-user persistent memory."""

    def __init__(self) -> None:
        self._memory = None

    def _get_memory(self):
        if self._memory is None:
            from mem0 import AsyncMemory
            config = {
                "llm": {"provider": "openai", "config": {"model": settings.MEM0_LLM_MODEL, "api_key": settings.OPENAI_API_KEY}},
                "embedder": {"provider": "openai", "config": {"model": settings.MEM0_EMBEDDER_MODEL, "api_key": settings.OPENAI_API_KEY}},
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
        """Add conversation turn to long-term memory."""
        try:
            memory = self._get_memory()
            await memory.add(messages, user_id=user_id)
            logger.info("ltm_memory_added", user_id=user_id)
        except Exception as exc:
            logger.warning("ltm_add_failed", user_id=user_id, error=str(exc))

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def search(self, user_id: str, query: str, limit: int = 5) -> List[str]:
        """Search for relevant memories for a given query."""
        try:
            memory = self._get_memory()
            results = await memory.search(query=query, user_id=user_id, limit=limit)
            return [r["memory"] for r in results.get("results", [])]
        except Exception as exc:
            logger.warning("ltm_search_failed", user_id=user_id, error=str(exc))
            return []

    async def delete_all(self, user_id: str) -> None:
        """Delete all memories for a user."""
        try:
            memory = self._get_memory()
            await memory.delete_all(user_id=user_id)
            logger.info("ltm_memories_deleted", user_id=user_id)
        except Exception as exc:
            logger.warning("ltm_delete_failed", user_id=user_id, error=str(exc))
