"""FastAPI application factory with lifespan, middleware, and router registration."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.api.v1 import auth, chat, eval, ingest
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import setup_logging
from app.core.middleware import CorrelationIdMiddleware, LatencyMiddleware

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown logic."""
    setup_logging()
    logger.info("app_starting", environment=settings.ENVIRONMENT.value, version=settings.VERSION)

    # Initialize Langfuse global client (required for @observe decorator inside CallbackHandler)
    if settings.langfuse_enabled:
        from langfuse import Langfuse
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info("langfuse_initialized", host=settings.LANGFUSE_HOST)

    # Initialize LangGraph checkpointer tables
    try:
        from app.memory.shortterm import setup_checkpointer
        await setup_checkpointer()
        logger.info("checkpointer_ready")
    except Exception as exc:
        logger.warning("checkpointer_setup_failed", error=str(exc))

    # Pre-warm embedding model
    try:
        from app.services.embedding import embedding_service
        await embedding_service.warmup()
        logger.info("embedding_model_ready", model=settings.EMBEDDING_MODEL)
    except Exception as exc:
        logger.warning("embedding_warmup_failed", error=str(exc))

    yield

    logger.info("app_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Production RAG Chatbot API",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── Rate limiter ──────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── Middleware (order matters: first added = outermost) ───────────────────
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(LatencyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Prometheus metrics endpoint ───────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Routers ───────────────────────────────────────────────────────────────
    prefix = settings.API_V1_STR
    app.include_router(auth.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(eval.router, prefix=prefix)

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health() -> dict:
        return {"status": "ok", "version": settings.VERSION, "env": settings.ENVIRONMENT.value}

    return app


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning("rate_limit_exceeded", path=request.url.path)
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


app = create_app()
