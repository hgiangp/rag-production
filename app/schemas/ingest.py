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
