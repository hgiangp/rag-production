"""Qdrant vector store client and collection management."""

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """Manages Qdrant client and collection lifecycle."""

    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY or None,
                prefer_grpc=settings.QDRANT_PREFER_GRPC,
                grpc_port=settings.QDRANT_GRPC_PORT,
            )
        return self._client

    async def ensure_collection(self, name: str, vector_size: int = 384) -> None:
        """Create collection if it doesn't exist."""
        try:
            existing = self.client.get_collections().collections
            if any(c.name == name for c in existing):
                return
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", collection=name, vector_size=vector_size)
        except Exception as exc:
            logger.exception("qdrant_collection_error", collection=name, error=str(exc))
            raise

    def collection_exists(self, name: str) -> bool:
        existing = self.client.get_collections().collections
        return any(c.name == name for c in existing)


vector_store_service = VectorStoreService()
