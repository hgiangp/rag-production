"""Cross-reference detection and retrieval for RAG.

Handles two reference types found in technical specification documents:

  Case 1 — Intra-document section refs:
    "see section 3.1.2.3", "refer to 4.2", "3.1.2.3項"
    → Fetch the target section from the SAME document (filter: document_id + section_number)
    → Fallback: semantic search within same document if section_number is mis-tagged by Dockling

  Case 2 — Inter-document spec refs:
    "refer to spec 'EnlargeWA'", "(EnlargeWA)", "refer to spec EnlargeWA"
    → Semantic search within the named spec (filter: spec_name) using the reference
      context as the query — returns the most relevant chunks, not arbitrary ones

Also handles: Table refs, Figure refs, Appendix refs, Requirement IDs.

Document filename convention:
    {model}_{(spec_name)}_{lang}_{version}.docx
    e.g.  781_(EnlargeWA)_E_250117.docx
          model=781, spec=EnlargeWA, lang=E, version=250117

Loop protection: visited sets prevent re-fetching the same section/spec within one
resolution call. Max depth is enforced per call (not recursive here — callers manage
depth if they chain calls).

Parallelism: all fetch tasks within a single resolution call are executed concurrently
via asyncio.gather.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Set

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings

logger = structlog.get_logger(__name__)


# ── Filename metadata ─────────────────────────────────────────────────────────

@dataclass
class DocumentMetadata:
    """Parsed metadata from a document filename."""

    model_symbol: str
    spec_name: str
    language: str   # "E" = English, "J" = Japanese
    version: str    # YYMMDD
    raw_filename: str

    @classmethod
    def from_filename(cls, filename: str) -> Optional["DocumentMetadata"]:
        """Parse '781_(EnlargeWA)_E_250117.docx' → DocumentMetadata."""
        name = filename.rsplit(".", 1)[0] if "." in filename else filename
        patterns = [
            r"^(\d+)_\(([^)]+)\)_([EJ])_(\d{6})$",   # 781_(EnlargeWA)_E_250117
            r"^(\d+)_([^_]+)_([EJ])_(\d{6})$",        # 781_EnlargeWA_E_250117
            r"^(.+?)_([^_]+)_([EJ])_(\d{6})$",        # anything_Spec_E_250117
        ]
        for pattern in patterns:
            m = re.match(pattern, name)
            if m:
                return cls(
                    model_symbol=m.group(1),
                    spec_name=m.group(2),
                    language=m.group(3),
                    version=m.group(4),
                    raw_filename=filename,
                )
        logger.debug("filename_parse_failed", filename=filename)
        return None


# ── Reference data classes ────────────────────────────────────────────────────

@dataclass
class SectionReference:
    section_number: str   # "3.1.2.3"
    original_text: str    # the matched string, for logging


@dataclass
class DocumentReference:
    spec_name: str        # "EnlargeWA"
    original_text: str
    model_symbol: Optional[str] = None


@dataclass
class CrossReference:
    section_refs: List[SectionReference] = field(default_factory=list)
    document_refs: List[DocumentReference] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.section_refs and not self.document_refs


# ── Detector ──────────────────────────────────────────────────────────────────

class CrossReferenceDetector:
    """Regex-based cross-reference detector for technical spec documents."""

    # ---- Section / clause patterns -----------------------------------------
    SECTION_PATTERNS = [
        # "see section 3.1.2.3", "refer to section 3", "Section 3.1"
        r"(?:see|refer to|as described in|detailed in|defined in|per|in|from)?\s*"
        r"(?:section|sect\.?|clause|§)\s*(\d+(?:\.\d+)*)",
        # Japanese clause markers: 3.1.2.3項 / 節 / 章
        r"(\d+(?:\.\d+)*)(?:項|節|章)",
        # Bare number in clear referential context: "in 3.1.2," or "per 3.1.2.3."
        r"(?:in|per|from|see)\s+(\d+\.\d+(?:\.\d+)*)\b",
    ]

    # ---- Document / spec patterns ------------------------------------------
    DOC_PATTERNS = [
        # refer to spec 'EnlargeWA' / "EnlargeWA" / `EnlargeWA` — quoted name
        r"(?:refer to|see|as per|defined in|per)\s+"
        r"(?:spec(?:ification)?|document|doc|仕様書?)\s*['\"`「『]([^'\"`」』]+)['\"`」』]",
        # refer to spec EnlargeWA — no quotes, keyword required
        r"(?:refer to|see|as per|defined in|per)\s+"
        r"(?:spec(?:ification)?|document|doc)\s+([A-Z][A-Za-z0-9_\-]{2,})\b",
        # 'EnlargeWA' specification  (quoted name before keyword)
        r"['\"`「『]([A-Za-z0-9_\-]+)['\"`」』]\s+(?:spec(?:ification)?|document|仕様)",
        # (EnlargeWA) — parenthesised spec name (alphanumeric, first letter upper)
        r"\(([A-Z][A-Za-z0-9]+(?:WA|Spec)?)\)",
        # 「EnlargeWA」 or 『EnlargeWA』 — Japanese quotation marks
        r"[「『]([A-Za-z0-9]+)[」』]",
        # EnlargeWA specification / EnlargeWA document
        r"\b([A-Z][A-Za-z0-9]{2,}(?:WA|Spec)?)\s+(?:spec(?:ification)?|document|仕様)",
    ]

    # ---- Anchor / label patterns (table, figure, appendix, requirement) ----
    ANCHOR_PATTERNS = [
        # Table 3-1, Figure 5.2, Appendix A, Appendix B.1
        r"(?:Table|Figure|Fig\.?|Appendix|App\.?)\s+([\w][\w.\-]*)",
        # Requirement IDs: R-001, REQ-042, SYS_REQ_005
        r"\b((?:R|REQ|SYS_REQ|SW_REQ)[-_]\d+)\b",
    ]

    # False positives to ignore for document refs
    _DOC_STOPWORDS = {
        "the", "this", "that", "see", "refer", "a", "an", "and", "or",
        "is", "are", "be", "been", "for", "to", "of", "in", "on",
    }

    def detect(self, text: str) -> CrossReference:
        return CrossReference(
            section_refs=self._detect_sections(text),
            document_refs=self._detect_documents(text),
        )

    def _detect_sections(self, text: str) -> List[SectionReference]:
        refs, seen = [], set()
        for pattern in self.SECTION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                num = m.group(1)
                if num not in seen:
                    seen.add(num)
                    refs.append(SectionReference(section_number=num, original_text=m.group(0).strip()))
        return refs

    def _detect_documents(self, text: str) -> List[DocumentReference]:
        refs, seen = [], set()
        for pattern in self.DOC_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                name = m.group(1).strip()
                if name.lower() in self._DOC_STOPWORDS or len(name) < 3:
                    continue
                if name not in seen:
                    seen.add(name)
                    refs.append(DocumentReference(spec_name=name, original_text=m.group(0).strip()))
        return refs

    def has_references(self, text: str) -> bool:
        """Quick check — avoids full detection when text is clean."""
        quick_patterns = [
            r"section\s+\d", r"\d+\.\d+(?:\.\d+)+", r"[項節章]",
            r"spec(?:ification)?\s*['\"`「『]", r"spec(?:ification)?\s+[A-Z]",
            r"\([A-Z][A-Za-z0-9]+\)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in quick_patterns)


# ── Retriever ─────────────────────────────────────────────────────────────────

class CrossReferenceRetriever:
    """Resolves cross-references from retrieved chunks using Qdrant payload filters
    and semantic vector search.

    Strategy:
      Section refs — try exact (document_id + section_number) filter first;
        fall back to semantic search within the same document when the section
        number is mis-tagged (e.g. Dockling numbering errors).
      Spec refs    — use semantic search filtered by spec_name so the most
        relevant chunks from the referenced spec are returned, not arbitrary ones.

    All fetch tasks within a resolution call run in parallel via asyncio.gather.

    Usage:
        retriever = CrossReferenceRetriever(aclient, embed_fn=_embed_query)
        extra = await retriever.resolve_from_nodes(nodes, collection)
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        embed_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
    ) -> None:
        self._client = qdrant_client
        self._detector = CrossReferenceDetector()
        self._embed = embed_fn

    async def resolve_from_nodes(
        self,
        nodes: List,
        collection: str,
        max_section_refs: int = 3,
        max_doc_refs: int = 3,
    ) -> List[str]:
        """Auto-resolve all cross-references found across a list of retrieved nodes.

        Each node must have metadata with at least 'document_id'.
        Loop-safe: visited sets prevent fetching the same target twice.
        All fetches run in parallel via asyncio.gather.

        Returns:
            Flat list of formatted context strings ready to append to the LLM prompt.
        """
        if not nodes:
            return []

        visited_sections: Set[str] = set()
        visited_docs: Set[str] = set()

        # Pre-populate visited with chunks we already have to avoid re-fetching them
        for node in nodes:
            doc_id = node.metadata.get("document_id", "")
            sec = node.metadata.get("section_number", "")
            spec = node.metadata.get("spec_name", "")
            if doc_id and sec:
                visited_sections.add(f"{doc_id}::{sec}")
            if spec:
                visited_docs.add(spec)

        col = _llama_collection(collection)
        fetch_tasks = []
        task_labels: List[tuple] = []  # (kind, label, section_or_spec)

        for node in nodes:
            text = getattr(node, "text", "") or ""
            if not text:
                continue

            has_refs = self._detector.has_references(text)
            logger.debug(
                "cross_ref_detector_check",
                has_references=has_refs,
                text_preview=text[:120].replace("\n", " "),
            )
            if not has_refs:
                continue

            refs = self._detector.detect(text)
            logger.debug(
                "cross_ref_detected",
                section_ref_count=len(refs.section_refs),
                doc_ref_count=len(refs.document_refs),
                section_numbers=[r.section_number for r in refs.section_refs],
                spec_names=[r.spec_name for r in refs.document_refs],
            )
            if refs.is_empty():
                continue

            doc_id = node.metadata.get("document_id", "")

            for ref in refs.section_refs[:max_section_refs]:
                key = f"{doc_id}::{ref.section_number}"
                if key in visited_sections:
                    continue
                visited_sections.add(key)
                fetch_tasks.append(
                    self._fetch_by_section(col, doc_id, ref.section_number, query_context=text)
                )
                task_labels.append(("section", ref.section_number))

            for ref in refs.document_refs[:max_doc_refs]:
                if ref.spec_name in visited_docs:
                    continue
                visited_docs.add(ref.spec_name)
                fetch_tasks.append(
                    self._fetch_by_spec(col, ref.spec_name, query_context=text)
                )
                task_labels.append(("spec", ref.spec_name))

        if not fetch_tasks:
            logger.info(
                "cross_references_resolved",
                collection=collection,
                node_count=len(nodes),
                extra_context_count=0,
            )
            return []

        # Run all fetches in parallel
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        extra_contexts: List[str] = []
        for (kind, label), result in zip(task_labels, results):
            if isinstance(result, Exception):
                logger.warning("cross_ref_fetch_failed", kind=kind, label=label, error=str(result))
                continue
            for chunk in result:
                if kind == "section":
                    extra_contexts.append(f"[Cross-ref: section {label} in same document]\n{chunk}")
                else:
                    extra_contexts.append(f"[Cross-ref: specification '{label}']\n{chunk}")

        logger.info(
            "cross_references_resolved",
            collection=collection,
            node_count=len(nodes),
            tasks_count=len(fetch_tasks),
            extra_context_count=len(extra_contexts),
        )
        return extra_contexts

    async def resolve_from_text(
        self,
        text: str,
        document_id: str,
        collection: str,
        max_section_refs: int = 3,
        max_doc_refs: int = 3,
    ) -> List[str]:
        """Resolve cross-references from a single text string.

        Used by the explicit `resolve_cross_references` agent tool when the agent
        has already identified a chunk that needs reference expansion.
        All fetches run in parallel via asyncio.gather.
        """
        has_refs = self._detector.has_references(text)
        logger.debug(
            "cross_ref_detector_check",
            has_references=has_refs,
            text_preview=text[:120].replace("\n", " "),
        )
        if not has_refs:
            return []

        refs = self._detector.detect(text)
        logger.debug(
            "cross_ref_detected",
            section_ref_count=len(refs.section_refs),
            doc_ref_count=len(refs.document_refs),
            section_numbers=[r.section_number for r in refs.section_refs],
            spec_names=[r.spec_name for r in refs.document_refs],
        )
        if refs.is_empty():
            return []

        col = _llama_collection(collection)
        fetch_tasks = []
        task_labels: List[tuple] = []

        for ref in refs.section_refs[:max_section_refs]:
            fetch_tasks.append(
                self._fetch_by_section(col, document_id, ref.section_number, query_context=text)
            )
            task_labels.append(("section", ref.section_number))

        for ref in refs.document_refs[:max_doc_refs]:
            fetch_tasks.append(
                self._fetch_by_spec(col, ref.spec_name, query_context=text)
            )
            task_labels.append(("spec", ref.spec_name))

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        extra: List[str] = []
        for (kind, label), result in zip(task_labels, results):
            if isinstance(result, Exception):
                logger.warning("cross_ref_fetch_failed", kind=kind, label=label, error=str(result))
                continue
            for chunk in result:
                if kind == "section":
                    extra.append(f"[Cross-ref: section {label}]\n{chunk}")
                else:
                    extra.append(f"[Cross-ref: spec '{label}']\n{chunk}")

        return extra

    # ── Qdrant helpers ────────────────────────────────────────────────────────

    async def _fetch_by_section(
        self,
        collection: str,
        document_id: str,
        section_number: str,
        limit: int = 2,
        query_context: str = "",
    ) -> List[str]:
        """Fetch section chunks with three-tier strategy:

        1. Exact filter: document_id + section_number
        2. Parent prefix: e.g. "3.1.2" when "3.1.2.3" has no match
        3. Semantic fallback: vector search within same document
           (handles Dockling mis-numbering — uses surrounding context as query)
        """
        if not document_id:
            return []
        try:
            # Tier 1: exact section filter
            chunks = await self._scroll_by_section(collection, document_id, section_number, limit)
            logger.debug(
                "section_filter_result",
                document_id=document_id,
                section=section_number,
                found=len(chunks),
                tier=1,
            )

            # Tier 2: parent prefix (e.g. "3.1.2" when "3.1.2.3" not tagged)
            if not chunks and "." in section_number:
                parent = section_number.rsplit(".", 1)[0]
                chunks = await self._scroll_by_section(collection, document_id, parent, limit)
                logger.debug(
                    "section_filter_result",
                    document_id=document_id,
                    section=parent,
                    found=len(chunks),
                    tier=2,
                )

            # Tier 3: semantic fallback — section tag may be wrong (Dockling numbering)
            if not chunks and query_context and self._embed:
                logger.debug(
                    "section_semantic_fallback_triggered",
                    document_id=document_id,
                    section=section_number,
                )
                vector = await self._embed(query_context)
                sem_results = await self._client.search(
                    collection_name=collection,
                    query_vector=vector,
                    query_filter=Filter(must=[
                        FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    ]),
                    limit=limit,
                    with_payload=True,
                )
                chunks = [
                    r.payload["text"]
                    for r in sem_results
                    if r.payload and "text" in r.payload
                ]
                logger.debug(
                    "section_filter_result",
                    document_id=document_id,
                    section=section_number,
                    found=len(chunks),
                    tier=3,
                )

            return chunks

        except Exception as exc:
            logger.warning("section_fetch_error", section=section_number, error=str(exc))
            return []

    async def _scroll_by_section(
        self, collection: str, document_id: str, section_number: str, limit: int
    ) -> List[str]:
        """Scroll filter: document_id + exact section_number."""
        results, _ = await self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="section_number", match=MatchValue(value=section_number)),
            ]),
            limit=limit,
            with_payload=True,
        )
        return [p.payload["text"] for p in results if p.payload and "text" in p.payload]

    async def _fetch_by_spec(
        self,
        collection: str,
        spec_name: str,
        limit: int = 3,
        query_context: str = "",
    ) -> List[str]:
        """Fetch chunks from a referenced specification.

        When query_context + embed_fn are available: semantic search filtered by
        spec_name — returns the chunks most relevant to the referencing text, not
        arbitrary first-N chunks.

        Falls back to scroll filter when no embedding function is provided.
        """
        try:
            if query_context and self._embed:
                # Semantic search: query = surrounding context, filter = spec_name
                vector = await self._embed(query_context)
                results = await self._client.search(
                    collection_name=collection,
                    query_vector=vector,
                    query_filter=Filter(must=[
                        FieldCondition(key="spec_name", match=MatchValue(value=spec_name)),
                    ]),
                    limit=limit,
                    with_payload=True,
                )
                chunks = []
                for r in results:
                    if r.payload and "text" in r.payload:
                        fname = r.payload.get("filename", "")
                        chunks.append(f"{r.payload['text']}\n[Source: {fname}]")
                logger.debug(
                    "spec_ref_fetched",
                    spec_name=spec_name,
                    found=len(chunks),
                    method="semantic",
                )
            else:
                # Fallback: scroll filter (no embedding available)
                results, _ = await self._client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="spec_name", match=MatchValue(value=spec_name)),
                    ]),
                    limit=limit,
                    with_payload=True,
                )
                chunks = []
                for p in results:
                    if p.payload and "text" in p.payload:
                        fname = p.payload.get("filename", "")
                        chunks.append(f"{p.payload['text']}\n[Source: {fname}]")
                logger.debug(
                    "spec_ref_fetched",
                    spec_name=spec_name,
                    found=len(chunks),
                    method="filter_scroll",
                )

            return chunks

        except Exception as exc:
            logger.warning("spec_fetch_error", spec_name=spec_name, error=str(exc))
            return []


# ── Metadata helpers (used at index time) ────────────────────────────────────

def extract_section_number(text: str) -> Optional[str]:
    """Extract leading section number from heading text.

    '3.1.2 Overview' → '3.1.2'
    'Section 4.5'    → '4.5'
    '1 Introduction' → '1'
    """
    patterns = [
        r"^(\d+(?:\.\d+)*)\s+",
        r"^(?:Section|§|Clause)\s*(\d+(?:\.\d+)*)",
    ]
    for pattern in patterns:
        m = re.match(pattern, text.strip(), re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def enrich_node_metadata(
    metadata: Dict,
    filename: str,
    heading_text: Optional[str] = None,
) -> Dict:
    """Add parsed filename fields and section_number to node metadata."""
    enriched = metadata.copy()
    doc_meta = DocumentMetadata.from_filename(filename)
    if doc_meta:
        enriched["model_symbol"] = doc_meta.model_symbol
        enriched["spec_name"] = doc_meta.spec_name
        enriched["language"] = doc_meta.language
        enriched["version"] = doc_meta.version
    if heading_text:
        sec = extract_section_number(heading_text)
        if sec:
            enriched["section_number"] = sec
    return enriched


def _llama_collection(collection: str) -> str:
    """Qdrant collection name for the LlamaIndex side."""
    return f"{collection}__llama"
