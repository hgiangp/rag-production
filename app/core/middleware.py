"""HTTP middleware: correlation ID injection and request latency recording."""

import time
import uuid
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = structlog.get_logger(__name__)

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Injects a correlation ID into every request.

    Reads from X-Correlation-ID header or generates a new UUID4.
    Binds the ID to structlog context so all log lines within the request carry it.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response


class LatencyMiddleware(BaseHTTPMiddleware):
    """Records per-request latency to Prometheus and adds X-Process-Time header."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Normalize path for Prometheus labels (avoid high cardinality)
        path = _normalize_path(request.url.path)
        status = str(response.status_code)

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=path,
            status_code=status,
        ).observe(duration)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=path,
            status_code=status,
        ).inc()

        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response


def _normalize_path(path: str) -> str:
    """Replace UUID/numeric path segments with placeholders to reduce cardinality."""
    import re

    # Replace UUIDs
    path = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "{id}", path)
    # Replace pure numeric IDs
    path = re.sub(r"/\d+", "/{id}", path)
    return path
