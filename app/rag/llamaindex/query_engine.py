"""LlamaIndex query engines: AutoMergingRetriever (primary) and RecursiveRetriever (secondary).

AutoMerging: retrieves leaf nodes from Qdrant; merges sibling leaves into parent when
  ≥ LLAMAINDEX_AUTO_MERGE_RATIO_THRESH of the parent's children are retrieved.
  Best for: dense factual questions where adjacent chunks form a coherent answer.

Recursive:   starts from retrieved leaf nodes and follows parent-child relationships
  upward through the docstore node graph.
  Best for: multi-hop questions that span several document sections.
"""

from typing import Literal

import structlog
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, RecursiveRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LlamaQueryEngine:
    """Wraps a VectorStoreIndex with AutoMergingRetriever or RecursiveRetriever."""

    def __init__(
        self,
        index: VectorStoreIndex,
        docstore: SimpleDocumentStore,
        mode: Literal["auto_merging", "recursive"] = "auto_merging",
    ) -> None:
        self._index = index
        self._docstore = docstore
        self._mode = mode
        self._engine = self._build_engine()

    def _build_engine(self) -> RetrieverQueryEngine:
        base_retriever = self._index.as_retriever(
            similarity_top_k=settings.LLAMAINDEX_SIMILARITY_TOP_K
        )

        if self._mode == "auto_merging":
            retriever = AutoMergingRetriever(
                base_retriever,
                self._index.storage_context,
                simple_ratio_thresh=settings.LLAMAINDEX_AUTO_MERGE_RATIO_THRESH,
                verbose=False,
            )
        else:  # recursive
            retriever = RecursiveRetriever(
                "vector",
                retriever_dict={"vector": base_retriever},
                node_dict=self._docstore.docs,
                verbose=False,
            )

        postprocessors = []
        if settings.reranking_enabled:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
            postprocessors.append(
                CohereRerank(
                    api_key=settings.COHERE_API_KEY,
                    model=settings.RERANKER_MODEL,
                    top_n=settings.RERANKER_TOP_N,
                )
            )

        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
        )

    async def query(self, query_str: str) -> str:
        """Execute async RAG query and return formatted result with source citations."""
        try:
            response = await self._engine.aquery(query_str)
            source_lines = [
                f"[Source: {n.metadata.get('filename', 'unknown')}]"
                for n in getattr(response, "source_nodes", [])
            ]
            result = str(response)
            if source_lines:
                result += "\n\n" + "\n".join(source_lines)
            return result
        except Exception as exc:
            logger.exception("llamaindex_query_failed", mode=self._mode, error=str(exc))
            return f"QUERY_ERROR: {str(exc)}"
