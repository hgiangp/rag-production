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
import difflib
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Literal, Optional, Set

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field as PydanticField
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings

logger = structlog.get_logger(__name__)


# ── LLM-based extractor ───────────────────────────────────────────────────────

class _CrossRefTarget(BaseModel):
    ref_type: Literal["section", "document"] = PydanticField(
        description=(
            "'section' for intra-document section/clause refs (e.g. '4.4.1.1', 'Section 3.2'); "
            "'document' for inter-document spec refs (e.g. 'EnlargeWA', 'SPEC \"Enlarge WA\"')"
        )
    )
    target: str = PydanticField(
        description=(
            "The target identifier: section number (e.g. '4.4.1.1') "
            "or spec/document name (e.g. 'EnlargeWA', 'Enlarge WA')"
        )
    )
    context_hint: str = PydanticField(
        description=(
            "Brief description of what the reference covers, derived from surrounding text "
            "or column headers — used as the semantic search query when fetching the target"
        )
    )


class _CrossRefExtraction(BaseModel):
    refs: List[_CrossRefTarget] = PydanticField(default_factory=list)


class LLMCrossReferenceExtractor:
    """LLM-based cross-reference extractor.

    Understands structured content (tables, lists) where regex keyword anchors
    are absent — e.g. bare '4.4.1.1' in a 'Reference' table column, or
    SPEC "Enlarge WA" as a standalone cell value.

    The regex CrossReferenceDetector.has_references() is still used as a fast
    boolean pre-screen before calling this extractor.

    Falls back gracefully: on LLM error, extract() returns an empty CrossReference
    so the caller continues without crashing.
    """

    _SYSTEM = (
        "You extract cross-references from technical specification document fragments.\n"
        "A cross-reference is any pointer to:\n"
        "  - A section or clause in the same document (ref_type='section'), "
        "e.g. '4.4.1.1', 'Section 3.2', '3.1.2.3項'\n"
        "  - A different specification document (ref_type='document'), "
        "e.g. 'EnlargeWA', 'SPEC \"Enlarge WA\"', '(EnlargeWA)'\n"
        "Include references that appear as bare numbers in table cells with no keyword context — "
        "use surrounding column headers or row labels to fill in 'context_hint'.\n"
        "Return an empty list when no cross-references are present."
    )

    def __init__(self, llm) -> None:
        self._structured_llm = llm.with_structured_output(_CrossRefExtraction)

    async def extract(self, text: str) -> "CrossReference":
        """Return a CrossReference extracted by the LLM. Returns empty on failure."""
        try:
            result: _CrossRefExtraction = await self._structured_llm.ainvoke([
                SystemMessage(content=self._SYSTEM),
                HumanMessage(content=f"Extract all cross-references:\n\n{text}"),
            ])
            return CrossReference(
                section_refs=[
                    SectionReference(section_number=r.target, original_text=r.context_hint)
                    for r in result.refs if r.ref_type == "section"
                ],
                document_refs=[
                    DocumentReference(spec_name=r.target, original_text=r.context_hint)
                    for r in result.refs if r.ref_type == "document"
                ],
            )
        except Exception as exc:
            logger.warning("llm_cross_ref_extraction_failed", error=str(exc))
            return CrossReference()


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

# Module-level spec_name cache: collection (with __llama suffix) → distinct stored spec_names.
# Populated on first resolution per collection; call invalidate_spec_name_cache() after re-indexing.
_spec_name_cache: Dict[str, List[str]] = {}


def _normalize_spec_name(name: str) -> str:
    """Lowercase and strip non-alphanumeric chars for fuzzy comparison.

    'Enlarge WA' → 'enlargewa'
    'enlagreWA'  → 'enlargewa'   (close enough for difflib tier-2)
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def invalidate_spec_name_cache(collection: str = "") -> None:
    """Clear cached spec_names. Call after re-indexing a collection."""
    if collection:
        _spec_name_cache.pop(_llama_collection(collection), None)
        _spec_name_cache.pop(collection, None)
    else:
        _spec_name_cache.clear()


async def ensure_payload_indexes(collection: str) -> None:
    """Create Qdrant payload indexes for cross-reference filtering fields.

    Indexes: document_id, spec_name, section_number, model_symbol, language.
    Safe to call repeatedly — ignores already-existing indexes.
    Called by the indexer after every document ingestion.
    """
    from qdrant_client.models import PayloadSchemaType

    aclient = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY or None,
    )
    col = _llama_collection(collection)
    fields = {
        "document_id": PayloadSchemaType.KEYWORD,
        "spec_name": PayloadSchemaType.KEYWORD,
        "section_number": PayloadSchemaType.KEYWORD,
        "model_symbol": PayloadSchemaType.KEYWORD,
        "language": PayloadSchemaType.KEYWORD,
    }
    for field, schema in fields.items():
        try:
            await aclient.create_payload_index(
                collection_name=col,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # index already exists
    logger.info("payload_indexes_ensured", collection=col)


class CrossReferenceRetriever:
    """Resolves structured CrossRefTarget dicts via Qdrant payload filters + semantic search.

    Called by the fetch_cross_ref_context graph node (primary) and by the
    resolve_cross_references agent tool (explicit single-chunk use).

    Spec name resolution — 3 tiers to handle typos / case / spacing variations:
      Tier 1: normalized exact match   ('Enlarge WA' → 'enlargewa' == stored 'enlargewa')
      Tier 2: fuzzy closest via difflib ('enlagreWA' ≈ 'enlargewa' at ≥0.75 cutoff)
      Tier 3: no spec filter — pure semantic search (last resort)

    Section resolution within a spec — 3 tiers for mis-tagged section numbers:
      Tier 1: exact  spec_name + section_number filter
      Tier 2: parent prefix  (strip last '.N' segment, e.g. '3.1.2.3' → '3.1.2')
      Tier 3: semantic within resolved spec_name (ignores section tag entirely)

    All fetches for a given resolve_targets() call run in parallel via asyncio.gather.
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        embed_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
    ) -> None:
        self._client = qdrant_client
        self._embed = embed_fn

    async def resolve_targets(
        self,
        targets: List[Dict],
        collection: str,
    ) -> List[str]:
        """Resolve a list of CrossRefTarget dicts to formatted context strings.

        Called by the fetch_cross_ref_context graph node and the
        resolve_cross_references agent tool.  All Qdrant fetches run in
        parallel; results are deduplicated by content hash.

        Args:
            targets:    List of CrossRefTarget dicts with keys:
                          spec_name (required), section_number (optional), query (required)
            collection: User-facing collection name (e.g. 'docs_usr_abc').
                        The __llama suffix is added internally.
        """
        if not targets:
            return []

        col = _llama_collection(collection)
        fetch_tasks = []
        labels: List[str] = []
        visited: Set[str] = set()

        for t in targets:
            spec = t.get("spec_name", "").strip()
            sec = t.get("section_number") or None
            if sec:
                sec = sec.strip() or None
            query = t.get("query", "")
            if not spec:
                continue
            key = f"{spec}::{sec or ''}"
            if key in visited:
                continue
            visited.add(key)

            if sec:
                fetch_tasks.append(self._fetch_by_spec_section(col, spec, sec, query))
                labels.append(f"section {sec} in {spec}")
            else:
                fetch_tasks.append(self._fetch_by_spec(col, spec, query))
                labels.append(f"spec {spec}")

        if not fetch_tasks:
            return []

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        contexts: List[str] = []
        seen_hashes: Set[int] = set()
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                logger.warning("cross_ref_fetch_failed", label=label, error=str(result))
                continue
            for chunk in result:
                h = hash(chunk.strip())
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    contexts.append(f"[Cross-ref: {label}]\n{chunk}")

        logger.info(
            "cross_ref_targets_resolved",
            collection=collection,
            target_count=len(targets),
            fetch_count=len(fetch_tasks),
            context_count=len(contexts),
        )
        return contexts

    # ── Qdrant helpers ────────────────────────────────────────────────────────

    async def _fetch_by_spec_section(
        self,
        collection: str,
        spec_name: str,
        section_number: str,
        query: str = "",
        limit: int = 2,
    ) -> List[str]:
        """Fetch a specific section within a spec — 3-tier fallback.

        Tier 1: fuzzy-resolved spec_name + exact section_number
        Tier 2: fuzzy-resolved spec_name + parent section prefix
        Tier 3: semantic search within fuzzy-resolved spec (drops section filter)
                If spec_name could not be resolved at all: semantic search unfiltered.
        """
        try:
            resolved = await self._resolve_spec_name(collection, spec_name)
            filter_name = resolved or spec_name
            unresolved = resolved is None

            # Tier 1: exact section within resolved spec
            chunks = await self._scroll_spec_section(collection, filter_name, section_number, limit)
            logger.debug(
                "spec_section_filter_result",
                spec=filter_name, section=section_number, found=len(chunks), tier=1,
            )

            # Tier 2: parent prefix (e.g. "3.1.2.3" → "3.1.2")
            if not chunks and "." in section_number:
                parent = section_number.rsplit(".", 1)[0]
                chunks = await self._scroll_spec_section(collection, filter_name, parent, limit)
                logger.debug(
                    "spec_section_filter_result",
                    spec=filter_name, section=parent, found=len(chunks), tier=2,
                )

            # Tier 3: semantic within spec (section tag may be wrong)
            if not chunks and query and self._embed:
                vector = await self._embed(query)
                q_filter = (
                    None if unresolved
                    else Filter(must=[FieldCondition(key="spec_name", match=MatchValue(value=filter_name))])
                )
                response = await self._client.query_points(
                    collection_name=collection,
                    query=vector,
                    query_filter=q_filter,
                    limit=limit,
                    with_payload=True,
                )
                chunks = [r.payload["text"] for r in response.points if r.payload and "text" in r.payload]
                logger.debug(
                    "spec_section_filter_result",
                    spec=filter_name, section=section_number, found=len(chunks), tier=3,
                    unfiltered=unresolved,
                )

            return chunks
        except Exception as exc:
            logger.warning("spec_section_fetch_error", spec=spec_name, section=section_number, error=str(exc))
            return []

    async def _scroll_spec_section(
        self, collection: str, spec_name: str, section_number: str, limit: int
    ) -> List[str]:
        """Scroll filter: spec_name + exact section_number."""
        results, _ = await self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="spec_name", match=MatchValue(value=spec_name)),
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
        query: str = "",
        limit: int = 3,
    ) -> List[str]:
        """Fetch chunks from a spec — fuzzy spec_name resolution then semantic search.

        Tier 1 + 2: _resolve_spec_name handles normalized-exact and fuzzy matching.
        Tier 3:     no spec filter — pure semantic search (when spec_name unresolvable).
        """
        try:
            resolved = await self._resolve_spec_name(collection, spec_name)

            if resolved and query and self._embed:
                vector = await self._embed(query)
                response = await self._client.query_points(
                    collection_name=collection,
                    query=vector,
                    query_filter=Filter(must=[
                        FieldCondition(key="spec_name", match=MatchValue(value=resolved))
                    ]),
                    limit=limit,
                    with_payload=True,
                )
                chunks = [
                    f"{r.payload['text']}\n[Source: {r.payload.get('filename', '')}]"
                    for r in response.points if r.payload and "text" in r.payload
                ]
                logger.debug("spec_ref_fetched", spec=resolved, found=len(chunks), method="semantic_resolved")

            elif resolved:
                results, _ = await self._client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="spec_name", match=MatchValue(value=resolved))
                    ]),
                    limit=limit,
                    with_payload=True,
                )
                chunks = [
                    f"{p.payload['text']}\n[Source: {p.payload.get('filename', '')}]"
                    for p in results if p.payload and "text" in p.payload
                ]
                logger.debug("spec_ref_fetched", spec=resolved, found=len(chunks), method="scroll_resolved")

            elif query and self._embed:
                # Tier 3: spec_name unresolvable — semantic search without filter
                vector = await self._embed(query)
                response = await self._client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                )
                chunks = [
                    f"{r.payload['text']}\n[Source: {r.payload.get('filename', '')}]"
                    for r in response.points if r.payload and "text" in r.payload
                ]
                logger.debug("spec_ref_fetched", spec=spec_name, found=len(chunks), method="semantic_unfiltered")

            else:
                logger.warning("spec_ref_no_fallback", spec=spec_name)
                chunks = []

            return chunks
        except Exception as exc:
            logger.warning("spec_fetch_error", spec=spec_name, error=str(exc))
            return []

    async def _resolve_spec_name(self, collection: str, raw_name: str) -> Optional[str]:
        """Map a raw spec_name to one that exists in the collection.

        Tier 1: normalized exact  — 'Enlarge WA' → 'enlargewa' == stored 'enlargewa'
        Tier 2: fuzzy closest     — 'enlagreWA'  ≈ 'enlargewa' (difflib cutoff 0.75)
        Returns None when no close-enough match found (caller falls back to tier-3 unfiltered).
        """
        norm_raw = _normalize_spec_name(raw_name)
        if not norm_raw:
            return None

        all_names = await self._fetch_all_spec_names(collection)
        norm_to_original: Dict[str, str] = {}
        for name in all_names:
            n = _normalize_spec_name(name)
            if n and n not in norm_to_original:
                norm_to_original[n] = name

        # Tier 1: normalized exact
        if norm_raw in norm_to_original:
            resolved = norm_to_original[norm_raw]
            logger.debug("spec_name_resolved", raw=raw_name, resolved=resolved, tier=1)
            return resolved

        # Tier 2: fuzzy closest
        matches = difflib.get_close_matches(norm_raw, norm_to_original.keys(), n=1, cutoff=0.75)
        if matches:
            resolved = norm_to_original[matches[0]]
            logger.debug("spec_name_resolved", raw=raw_name, resolved=resolved, tier=2, matched=matches[0])
            return resolved

        logger.debug("spec_name_unresolved", raw=raw_name, known_count=len(norm_to_original))
        return None

    async def _fetch_all_spec_names(self, collection: str) -> List[str]:
        """Collect all distinct spec_name values stored in a collection. Cached."""
        if collection in _spec_name_cache:
            return _spec_name_cache[collection]

        spec_names: Set[str] = set()
        offset = None
        while True:
            results, offset = await self._client.scroll(
                collection_name=collection,
                limit=250,
                with_payload=["spec_name"],
                offset=offset,
            )
            for p in results:
                if p.payload and p.payload.get("spec_name"):
                    spec_names.add(p.payload["spec_name"])
            if offset is None:
                break

        names = list(spec_names)
        _spec_name_cache[collection] = names
        logger.debug("spec_names_cached", collection=collection, count=len(names))
        return names


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
