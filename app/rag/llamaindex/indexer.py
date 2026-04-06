"""LlamaIndex indexing pipeline: sentence-window and auto-merging strategies."""

import structlog
from llama_index.core import Document, Settings as LISettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    get_leaf_nodes,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LlamaIndexer:
    """Manages LlamaIndex-specific index construction.

    Two strategies:
    1. sentence_window — stores each sentence with surrounding window for context
    2. auto_merging — hierarchical chunks that auto-merge on retrieval
    """

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )

    async def build_sentence_window_index(
        self, documents: list[Document], collection: str
    ) -> VectorStoreIndex:
        """Build sentence-window index — stores sentences with ±N sentence window."""
        from app.services.embedding import embedding_service
        from app.services.llm import get_llm

        LISettings.llm = get_llm(model=settings.DEFAULT_LLM_MODEL, framework="llamaindex")
        LISettings.embed_model = embedding_service.llamaindex_model

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=settings.LLAMAINDEX_SENTENCE_WINDOW_SIZE,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        vector_store = QdrantVectorStore(
            client=self._client, collection_name=f"{collection}__llama_sw"
        )
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_ctx)
        logger.info("sentence_window_index_built", node_count=len(nodes), collection=collection)
        return index

    async def build_auto_merging_index(
        self, documents: list[Document], collection: str
    ) -> VectorStoreIndex:
        """Build auto-merging (hierarchical) index — merges retrieved leaf nodes into parents."""
        from app.services.embedding import embedding_service
        from app.services.llm import get_llm

        LISettings.llm = get_llm(model=settings.DEFAULT_LLM_MODEL, framework="llamaindex")
        LISettings.embed_model = embedding_service.llamaindex_model

        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=settings.LLAMAINDEX_AUTO_MERGE_CHUNK_SIZES
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)

        vector_store = QdrantVectorStore(
            client=self._client, collection_name=f"{collection}__llama_am"
        )
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store, docstore_nodes=nodes)
        index = VectorStoreIndex(leaf_nodes, storage_context=storage_ctx)
        logger.info("auto_merging_index_built", leaf_count=len(leaf_nodes), collection=collection)
        return index
