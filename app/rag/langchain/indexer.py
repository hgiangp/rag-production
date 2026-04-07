"""Hierarchical document indexer: parent (header-based) + child (fixed-size) chunks → Qdrant."""

from typing import List
from uuid import UUID

import structlog
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from app.core.config import settings
from app.core.metrics import INGEST_CHUNKS
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store_service

logger = structlog.get_logger(__name__)

_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3")]


class HierarchicalIndexer:
    """Splits documents into parent chunks (header boundaries) and child chunks (fixed-size).

    Parent chunks are stored in Qdrant with a parent_id metadata field.
    Child chunks are stored in a separate collection with a reference to their parent.
    """

    async def index_document(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        file_type: str,
        collection: str,
    ) -> int:
        """Parse, chunk, embed, and store a document. Returns total chunk count."""
        text = await self._extract_text(content, file_type, filename)
        parent_docs, child_docs = self._split_hierarchically(
            text=text, document_id=document_id, filename=filename
        )

        child_collection = f"{collection}__{settings.QDRANT_CHILD_COLLECTION}"
        parent_collection = f"{collection}__{settings.QDRANT_PARENT_COLLECTION}"

        await self._store_in_qdrant(child_docs, child_collection)
        await self._store_parents_raw(parent_docs, parent_collection)

        total = len(child_docs) + len(parent_docs)
        INGEST_CHUNKS.observe(total)
        logger.info(
            "document_indexed",
            document_id=document_id,
            parent_count=len(parent_docs),
            child_count=len(child_docs),
        )
        return total

    async def delete_document(self, document_id: str, collection: str) -> None:
        """Delete all chunks for a document from Qdrant."""
        client = vector_store_service.client
        for suffix in [settings.QDRANT_CHILD_COLLECTION, settings.QDRANT_PARENT_COLLECTION]:
            coll = f"{collection}__{suffix}"
            await client.delete(
                collection_name=coll,
                points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}},
            )
        logger.info("document_deleted_from_qdrant", document_id=document_id)

    def _split_hierarchically(
        self, text: str, document_id: str, filename: str
    ) -> tuple[List[Document], List[Document]]:
        """Split text into parent (header) and child (fixed-size) chunks."""
        # Parent: split on markdown headers
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
        parent_docs = md_splitter.split_text(text)

        # Fall back to fixed-size if no headers
        if not parent_docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.PARENT_CHUNK_SIZE,
                chunk_overlap=settings.CHILD_CHUNK_OVERLAP * 3,
            )
            parent_docs = splitter.create_documents([text])

        # Enrich parent metadata
        for i, doc in enumerate(parent_docs):
            doc.metadata.update({"document_id": document_id, "filename": filename, "parent_id": f"{document_id}_p{i}"})

        # Child: split each parent into small chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
        )
        child_docs: List[Document] = []
        for parent in parent_docs:
            children = child_splitter.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata["chunk_id"] = f"{parent.metadata['parent_id']}_c{j}"
            child_docs.extend(children)

        return parent_docs, child_docs

    async def _store_in_qdrant(self, docs: List[Document], collection: str) -> None:
        """Embed and upsert documents into Qdrant."""
        embeddings = embedding_service.model
        await vector_store_service.ensure_collection(collection)
        qs = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            collection_name=collection,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
        )

    async def _store_parents_raw(self, docs: List[Document], collection: str) -> None:
        """Store parent documents as-is (no embedding needed for retrieval by ID)."""
        client = vector_store_service.client
        await vector_store_service.ensure_collection(collection, vector_size=1)
        # Store parents with a dummy vector — retrieved by filter, not ANN
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=abs(hash(doc.metadata["parent_id"])) % (2**63),
                vector=[1.0],  # dummy; parents are fetched by filter, not ANN
                payload={"content": doc.page_content, **doc.metadata},
            )
            for doc in docs
        ]
        await client.upsert(collection_name=collection, points=points)

    @staticmethod
    async def _extract_text(content: bytes, file_type: str, filename: str) -> str:
        """Extract plain text from binary file content."""
        import io
        if file_type == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        if file_type == "docx":
            from docx import Document as DocxDocument
            doc = DocxDocument(io.BytesIO(content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        # txt / md
        return content.decode("utf-8", errors="replace")
