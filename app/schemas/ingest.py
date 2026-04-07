"""Document ingestion schemas."""

from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    status: Literal["pending", "processing", "indexed", "failed"]
    chunk_count: int = 0
    message: str = ""


class DocumentStatus(BaseModel):
    document_id: UUID
    filename: str
    status: Literal["pending", "processing", "indexed", "failed"]
    chunk_count: int
    error_msg: Optional[str] = None


class DeleteDocumentResponse(BaseModel):
    document_id: UUID
    deleted: bool


class BatchIngestResponse(BaseModel):
    """Response for batch document ingestion."""

    documents: List[IngestResponse]
    total_accepted: int
    message: str = ""


class BatchDeleteResponse(BaseModel):
    """Response for batch document deletion."""

    document_ids: List[UUID]
    deleted_count: int
