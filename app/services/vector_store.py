"""Qdrant vector store client and collection management."""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings
from app.services.embedding import embedding_service

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """Manages Qdrant client and collection lifecycle."""

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY or None,
                prefer_grpc=settings.QDRANT_PREFER_GRPC,
                grpc_port=settings.QDRANT_GRPC_PORT,
                timeout=10.0,
            )
        return self._client

    async def ensure_collection(self, name: str, vector_size: int | None = None) -> None:
        """Create collection if it doesn't exist, or recreate if vector size mismatch."""
        if vector_size is None:
            vector_size = embedding_service.get_vector_size()
        try:
            existing = (await self.client.get_collections()).collections
            collection_exists = any(c.name == name for c in existing)
            if collection_exists:
                collection_info = await self.client.get_collection(name)
                current_size = collection_info.config.params.vectors.size
                if current_size == vector_size:
                    return
                else:
                    logger.warning(
                        "qdrant_collection_size_mismatch",
                        collection=name,
                        current_size=current_size,
                        expected_size=vector_size,
                        action="recreating"
                    )
                    await self.client.delete_collection(name)
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", collection=name, vector_size=vector_size)
        except Exception as exc:
            logger.exception("qdrant_collection_error", collection=name, error=str(exc))
            raise

    async def collection_exists(self, name: str) -> bool:
        existing = (await self.client.get_collections()).collections
        return any(c.name == name for c in existing)


vector_store_service = VectorStoreService()
