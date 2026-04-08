"""Cross-reference detection and retrieval for RAG.

Handles two reference types found in technical specification documents:

  Case 1 — Intra-document section refs:
    "see section 3.1.2.3", "refer to 4.2", "3.1.2.3項"
    → Fetch the target section from the SAME document (filter: document_id + section_number)

  Case 2 — Inter-document spec refs:
    "refer to spec 'EnlargeWA'", "(EnlargeWA)"
    → Fetch intro/summary chunks from the named spec (filter: spec_name)

Also handles: Table refs, Figure refs, Appendix refs, Requirement IDs.

Document filename convention:
    {model}_{(spec_name)}_{lang}_{version}.docx
    e.g.  781_(EnlargeWA)_E_250117.docx
          model=781, spec=EnlargeWA, lang=E, version=250117

Loop protection: visited sets prevent re-fetching the same section/spec within one
resolution call. Max depth is enforced per call (not recursive here — callers manage
depth if they chain calls).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
        # refer to spec 'EnlargeWA' / "EnlargeWA" / `EnlargeWA`
        r"(?:refer to|see|as per|defined in|per)\s+"
        r"(?:spec(?:ification)?|document|doc|仕様書?)\s*['\"`「『]([^'\"`」』]+)['\"`」』]",
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
            r"spec(?:ification)?['\"`「]", r"\([A-Z][A-Za-z0-9]+\)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in quick_patterns)


# ── Retriever ─────────────────────────────────────────────────────────────────

class CrossReferenceRetriever:
    """Resolves cross-references from retrieved chunks using Qdrant payload filters.

    Usage:
        retriever = CrossReferenceRetriever(aclient)
        extra = await retriever.resolve_from_nodes(nodes, collection)
    """

    def __init__(self, qdrant_client: AsyncQdrantClient) -> None:
        self._client = qdrant_client
        self._detector = CrossReferenceDetector()

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

        Returns:
            Flat list of formatted context strings ready to append to the LLM prompt.
        """
        if not nodes:
            return []

        visited_sections: Set[str] = set()   # "doc_id::section_number"
        visited_docs: Set[str] = set()        # spec_name

        # Pre-populate visited with the chunks we already have, so we don't re-fetch them
        for node in nodes:
            doc_id = node.metadata.get("document_id", "")
            sec = node.metadata.get("section_number", "")
            spec = node.metadata.get("spec_name", "")
            if doc_id and sec:
                visited_sections.add(f"{doc_id}::{sec}")
            if spec:
                visited_docs.add(spec)

        extra_contexts: List[str] = []
        col = _llama_collection(collection)

        for node in nodes:
            text = getattr(node, "text", "") or ""
            if not text or not self._detector.has_references(text):
                continue

            refs = self._detector.detect(text)
            if refs.is_empty():
                continue

            doc_id = node.metadata.get("document_id", "")

            # ── Intra-document section references ────────────────────────────
            for ref in refs.section_refs[:max_section_refs]:
                key = f"{doc_id}::{ref.section_number}"
                if key in visited_sections:
                    continue
                visited_sections.add(key)

                chunks = await self._fetch_by_section(col, doc_id, ref.section_number)
                for chunk in chunks:
                    extra_contexts.append(
                        f"[Cross-ref: section {ref.section_number} in same document]\n{chunk}"
                    )

            # ── Inter-document spec references ───────────────────────────────
            for ref in refs.document_refs[:max_doc_refs]:
                if ref.spec_name in visited_docs:
                    continue
                visited_docs.add(ref.spec_name)

                chunks = await self._fetch_by_spec(col, ref.spec_name)
                for chunk in chunks:
                    extra_contexts.append(
                        f"[Cross-ref: specification '{ref.spec_name}']\n{chunk}"
                    )

        logger.info(
            "cross_references_resolved",
            collection=collection,
            node_count=len(nodes),
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
        """
        if not self._detector.has_references(text):
            return []

        refs = self._detector.detect(text)
        if refs.is_empty():
            return []

        col = _llama_collection(collection)
        extra: List[str] = []

        for ref in refs.section_refs[:max_section_refs]:
            chunks = await self._fetch_by_section(col, document_id, ref.section_number)
            extra.extend(
                f"[Cross-ref: section {ref.section_number}]\n{c}" for c in chunks
            )

        for ref in refs.document_refs[:max_doc_refs]:
            chunks = await self._fetch_by_spec(col, ref.spec_name)
            extra.extend(
                f"[Cross-ref: spec '{ref.spec_name}']\n{c}" for c in chunks
            )

        return extra

    # ── Qdrant helpers ────────────────────────────────────────────────────────

    async def _fetch_by_section(
        self, collection: str, document_id: str, section_number: str, limit: int = 2
    ) -> List[str]:
        """Filter: same document_id + exact section_number match."""
        if not document_id:
            return []
        try:
            results, _ = await self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="section_number", match=MatchValue(value=section_number)),
                ]),
                limit=limit,
                with_payload=True,
            )

            chunks = [p.payload["text"] for p in results if p.payload and "text" in p.payload]

            # Fallback: section may not have been tagged — search by prefix
            # (e.g. stored as "3.1.2" but referenced as "3.1.2.3")
            if not chunks and "." in section_number:
                parent_section = section_number.rsplit(".", 1)[0]
                chunks = await self._fetch_by_section(collection, document_id, parent_section, limit)

            logger.debug(
                "section_ref_fetched",
                document_id=document_id,
                section=section_number,
                found=len(chunks),
            )
            return chunks

        except Exception as exc:
            logger.warning("section_fetch_error", section=section_number, error=str(exc))
            return []

    async def _fetch_by_spec(
        self, collection: str, spec_name: str, limit: int = 3
    ) -> List[str]:
        """Filter: spec_name matches the referenced specification name."""
        try:
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

            logger.debug("spec_ref_fetched", spec_name=spec_name, found=len(chunks))
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
