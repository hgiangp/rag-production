"""Document ingestion endpoint."""

import asyncio
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.auth import CurrentUser, get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.core.metrics import INGEST_COUNT, INGEST_LATENCY
from app.rag.langchain.indexer import HierarchicalIndexer
from app.schemas.ingest import DeleteDocumentResponse, IngestResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["ingest"])

_indexer = HierarchicalIndexer()


@router.post("", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("10/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
) -> IngestResponse:
    """Upload and index a document. Processing happens asynchronously."""
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' not supported. Allowed: {settings.ALLOWED_FILE_TYPES}",
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
        )

    document_id = uuid4()
    collection = f"docs_{current_user.user_id}"

    logger.info(
        "ingest_started",
        document_id=str(document_id),
        filename=file.filename,
        size_bytes=len(content),
        collection=collection,
    )

    # Kick off indexing without blocking response (202 Accepted pattern)
    asyncio.create_task(
        _run_indexing(
            document_id=document_id,
            filename=file.filename or "unknown",
            content=content,
            file_type=ext,
            collection=collection,
            user_id=str(current_user.user_id),
        )
    )

    return IngestResponse(
        document_id=document_id,
        filename=file.filename or "unknown",
        status="processing",
        message="Document accepted for indexing",
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
) -> DeleteDocumentResponse:
    """Remove a document and all its chunks from the vector store."""
    collection = f"docs_{current_user.user_id}"
    try:
        await _indexer.delete_document(document_id=str(document_id), collection=collection)
        logger.info("document_deleted", document_id=str(document_id))
        return DeleteDocumentResponse(document_id=document_id, deleted=True)
    except Exception as exc:
        logger.exception("document_delete_failed", document_id=str(document_id), error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Delete failed")


async def _run_indexing(
    document_id: UUID,
    filename: str,
    content: bytes,
    file_type: str,
    collection: str,
    user_id: str,
) -> None:
    import time
    start = time.perf_counter()
    try:
        chunk_count = await _indexer.index_document(
            document_id=str(document_id),
            filename=filename,
            content=content,
            file_type=file_type,
            collection=collection,
        )
        duration = time.perf_counter() - start
        INGEST_COUNT.labels(file_type=file_type, status="success").inc()
        INGEST_LATENCY.labels(file_type=file_type).observe(duration)
        logger.info("ingest_completed", document_id=str(document_id), chunk_count=chunk_count)
    except Exception as exc:
        INGEST_COUNT.labels(file_type=file_type, status="failed").inc()
        logger.exception("ingest_failed", document_id=str(document_id), error=str(exc))
