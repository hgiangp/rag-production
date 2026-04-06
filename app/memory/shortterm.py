"""Short-term memory: LangGraph AsyncPostgresSaver for session checkpointing."""

from urllib.parse import quote_plus

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = structlog.get_logger(__name__)

_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        pw = quote_plus(settings.POSTGRES_PASSWORD)
        conn_str = (
            f"postgresql://{settings.POSTGRES_USER}:{pw}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        _pool = AsyncConnectionPool(conninfo=conn_str, max_size=settings.POSTGRES_POOL_SIZE, open=False)
        await _pool.open()
        logger.info("pg_connection_pool_opened", max_size=settings.POSTGRES_POOL_SIZE)
    return _pool


async def setup_checkpointer() -> AsyncPostgresSaver:
    """Initialize checkpointer tables in PostgreSQL."""
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(conn=pool)
    await checkpointer.setup()
    logger.info("langgraph_checkpointer_tables_ready")
    return checkpointer


async def get_checkpointer() -> AsyncPostgresSaver:
    """Return a ready checkpointer instance."""
    pool = await get_pool()
    return AsyncPostgresSaver(conn=pool)
