"""Document ingestion endpoint."""

import asyncio
from typing import List
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.auth import CurrentUser, get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.core.metrics import INGEST_COUNT, INGEST_LATENCY
from app.rag.llamaindex.indexer import DocumentInput, LlamaIndexer
from app.rag.llamaindex.cross_reference import ensure_payload_indexes
from app.rag.llamaindex.tools import invalidate_engine_cache
from app.schemas.ingest import BatchDeleteResponse, BatchIngestResponse, DeleteDocumentResponse, IngestResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["ingest"])

_indexer = LlamaIndexer()


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

    # Kick off indexing without blocking response (202 Accepted pattern — R7)
    asyncio.create_task(
        _run_indexing(
            document_id=document_id,
            filename=file.filename or "unknown",
            content=content,
            file_type=ext,
            collection=collection,
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
    """Remove a document and all its nodes from the vector store and docstore."""
    collection = f"docs_{current_user.user_id}"
    try:
        await _indexer.delete_document(document_id=str(document_id), collection=collection)
        invalidate_engine_cache(collection)
        logger.info("document_deleted", document_id=str(document_id))
        return DeleteDocumentResponse(document_id=document_id, deleted=True)
    except Exception as exc:
        logger.exception("document_delete_failed", document_id=str(document_id), error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Delete failed")


@router.post("/batch", response_model=BatchIngestResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("5/minute")
async def ingest_documents_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: CurrentUser = Depends(get_current_user),
) -> BatchIngestResponse:
    """Upload and index multiple documents in a single batch.

    More efficient than multiple single uploads:
    - Single docstore load/persist
    - Parallel document parsing
    - Single Qdrant batch insert
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    if len(files) > 20:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 20 files per batch")

    collection = f"docs_{current_user.user_id}"
    responses: List[IngestResponse] = []
    documents_to_index: List[DocumentInput] = []
    doc_id_map: dict[str, tuple[UUID, str, str]] = {}  # doc_id -> (uuid, filename, ext)

    # Validate and prepare all files
    for file in files:
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in settings.ALLOWED_FILE_TYPES:
            responses.append(
                IngestResponse(
                    document_id=uuid4(),
                    filename=file.filename or "unknown",
                    status="failed",
                    message=f"File type '{ext}' not supported",
                )
            )
            continue

        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            responses.append(
                IngestResponse(
                    document_id=uuid4(),
                    filename=file.filename or "unknown",
                    status="failed",
                    message=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
                )
            )
            continue

        document_id = uuid4()
        doc_id_str = str(document_id)
        documents_to_index.append(
            DocumentInput(
                document_id=doc_id_str,
                filename=file.filename or "unknown",
                content=content,
                file_type=ext,
            )
        )
        doc_id_map[doc_id_str] = (document_id, file.filename or "unknown", ext)
        responses.append(
            IngestResponse(
                document_id=document_id,
                filename=file.filename or "unknown",
                status="processing",
                message="Document accepted for indexing",
            )
        )

    if documents_to_index:
        logger.info(
            "batch_ingest_started",
            collection=collection,
            doc_count=len(documents_to_index),
        )

        # Kick off batch indexing without blocking response
        asyncio.create_task(
            _run_batch_indexing(
                documents=documents_to_index,
                collection=collection,
                doc_id_map=doc_id_map,
            )
        )

    accepted_count = len(documents_to_index)
    return BatchIngestResponse(
        documents=responses,
        total_accepted=accepted_count,
        message=f"{accepted_count} document(s) accepted for indexing",
    )


@router.delete("/batch", response_model=BatchDeleteResponse)
async def delete_documents_batch(
    request: Request,
    document_ids: List[UUID],
    current_user: CurrentUser = Depends(get_current_user),
) -> BatchDeleteResponse:
    """Delete multiple documents in a single batch."""
    if not document_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No document IDs provided")

    if len(document_ids) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 50 documents per batch")

    collection = f"docs_{current_user.user_id}"
    try:
        deleted_count = await _indexer.delete_documents(
            document_ids=[str(doc_id) for doc_id in document_ids],
            collection=collection,
        )
        invalidate_engine_cache(collection)
        logger.info("batch_documents_deleted", doc_count=deleted_count)
        return BatchDeleteResponse(document_ids=document_ids, deleted_count=deleted_count)
    except Exception as exc:
        logger.exception("batch_delete_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch delete failed")


async def _run_indexing(
    document_id: UUID,
    filename: str,
    content: bytes,
    file_type: str,
    collection: str,
) -> None:
    import time
    start = time.perf_counter()
    try:
        node_count = await _indexer.index_document(
            document_id=str(document_id),
            filename=filename,
            content=content,
            file_type=file_type,
            collection=collection,
        )
        duration = time.perf_counter() - start
        # Ensure payload indexes exist for cross-reference filtering
        await ensure_payload_indexes(collection)
        # Invalidate stale engine caches so next query loads fresh nodes
        invalidate_engine_cache(collection)
        INGEST_COUNT.labels(file_type=file_type, status="success").inc()
        INGEST_LATENCY.labels(file_type=file_type).observe(duration)
        logger.info("ingest_completed", document_id=str(document_id), node_count=node_count)
    except Exception as exc:
        INGEST_COUNT.labels(file_type=file_type, status="failed").inc()
        logger.exception("ingest_failed", document_id=str(document_id), error=str(exc))


async def _run_batch_indexing(
    documents: List[DocumentInput],
    collection: str,
    doc_id_map: dict[str, tuple[UUID, str, str]],
) -> None:
    """Background task for batch indexing."""
    import time
    start = time.perf_counter()
    try:
        node_counts = await _indexer.index_documents(documents=documents, collection=collection)
        duration = time.perf_counter() - start

        # Ensure payload indexes exist for cross-reference filtering
        await ensure_payload_indexes(collection)
        # Invalidate stale engine caches
        invalidate_engine_cache(collection)

        # Record metrics per file type
        for doc_id, count in node_counts.items():
            _, _, ext = doc_id_map.get(doc_id, (None, None, "unknown"))
            status = "success" if count > 0 else "failed"
            INGEST_COUNT.labels(file_type=ext, status=status).inc()

        INGEST_LATENCY.labels(file_type="batch").observe(duration)
        logger.info(
            "batch_ingest_completed",
            collection=collection,
            doc_count=len(documents),
            total_nodes=sum(node_counts.values()),
            duration=round(duration, 2),
        )
    except Exception as exc:
        # Record all as failed
        for doc_id in doc_id_map:
            _, _, ext = doc_id_map[doc_id]
            INGEST_COUNT.labels(file_type=ext, status="failed").inc()
        logger.exception("batch_ingest_failed", error=str(exc))
