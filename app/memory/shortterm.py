"""Short-term memory: LangGraph AsyncPostgresSaver for session checkpointing."""

from urllib.parse import quote_plus

import psycopg
import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = structlog.get_logger(__name__)

_pool: AsyncConnectionPool | None = None


def _build_conn_str() -> str:
    pw = quote_plus(settings.POSTGRES_PASSWORD)
    return (
        f"postgresql://{settings.POSTGRES_USER}:{pw}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(conninfo=_build_conn_str(), max_size=settings.POSTGRES_POOL_SIZE, open=False)
        await _pool.open()
        logger.info("pg_connection_pool_opened", max_size=settings.POSTGRES_POOL_SIZE)
    return _pool


async def setup_checkpointer() -> None:
    """Initialize checkpointer tables in PostgreSQL.

    Uses a direct autocommit connection because CREATE INDEX CONCURRENTLY
    (run internally by AsyncPostgresSaver.setup()) cannot execute inside a
    transaction block.  The pool used by get_checkpointer() is unaffected.
    """
    async with await psycopg.AsyncConnection.connect(_build_conn_str(), autocommit=True) as conn:
        await AsyncPostgresSaver(conn=conn).setup()
    logger.info("langgraph_checkpointer_tables_ready")


async def get_checkpointer() -> AsyncPostgresSaver:
    """Return a ready checkpointer instance."""
    pool = await get_pool()
    return AsyncPostgresSaver(conn=pool)
