"""Retrieval nodes: token estimation, context compression, cross-reference resolution."""

from typing import List, Optional

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from pydantic import Field as PydanticField
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.metrics import GRAPH_NODE_COUNT, GRAPH_NODE_LATENCY
from app.graph.state import AgentState, CrossRefTarget
from app.rag.llamaindex.cross_reference import (
    CrossReferenceDetector,
    CrossReferenceRetriever,
    LLMCrossReferenceExtractor,
)

logger = structlog.get_logger(__name__)

# Tool names whose results may contain cross-references.
_RAG_TOOL_NAMES = {"llamaindex_query", "llamaindex_recursive", "search_child_chunks", "fetch_parent_chunks"}

# ── Pydantic models for LLM structured extraction ────────────────────────────

class _CrossRefTargetLLM(BaseModel):
    spec_name: str = PydanticField(
        description=(
            "SHORT spec name — never the full filename. "
            "For intra-document section refs: copy the value from [Source: spec=<name>] exactly. "
            "For inter-document refs like 'SPEC \"Enlarge WA\"' or '(EnlargeWA)': extract only the name ('Enlarge WA'). "
            "If no spec= label is present, parse from the filename: '7821_(Pop-up)_E_210617.docx' → 'Pop-up'."
        )
    )
    section_number: Optional[str] = PydanticField(
        None,
        description=(
            "Dotted section number ONLY — e.g. '4.4.1.3', '3.1.2'. "
            "Set this when the reference points to a specific clause or section. "
            "Leave null for spec-level references (e.g. 'SPEC \"Enlarge WA\"' with no section)."
        ),
    )
    query: str = PydanticField(
        description=(
            "Short description of what the referenced content covers, used as a semantic "
            "search query against the document collection. "
            "IMPORTANT: Write this in the SAME language as the document text — "
            "Japanese for Japanese docs (e.g. 'アイテムバックボタンの表示仕様', 'スクロールバーの動作'), "
            "English for English docs (e.g. 'display specification of Item Back button'). "
            "Do NOT put section numbers or spec names here."
        ),
    )


class _CrossRefDetectionOutput(BaseModel):
    targets: List[_CrossRefTargetLLM] = PydanticField(default_factory=list)


_DETECT_SYSTEM = (
    "Extract cross-references from retrieved document chunks.\n"
    "Documents may be in English or Japanese — handle both equally.\n\n"
    "Each chunk is prefixed with: [Source: spec=<spec_name> | section=<sec> | file=<filename>]\n\n"
    "FIELD RULES:\n"
    "  spec_name     Short spec name from the [Source: spec=...] label, or parsed from the\n"
    "                filename ('7821_(Pop-up)_E_210617.docx' → 'Pop-up',\n"
    "                          '7821_(Pop-up)_J_210617.docx' → 'Pop-up').\n"
    "                For inter-document refs ('SPEC \"Enlarge WA\"', '（EnlargeWA）', '「EnlargeWA」'): "
    "use the referenced spec name.\n"
    "                NEVER copy the full filename into spec_name.\n"
    "  section_number  The dotted number (e.g. '4.4.1.3', '3.1.2.3'). Set only when ref is to a\n"
    "                  specific clause (including Japanese patterns like '3.1.2.3項'). "
    "Null for spec-level refs.\n"
    "  query         Brief description of what the reference covers — used as a semantic search\n"
    "                query against the SAME document collection.\n"
    "                WRITE IN THE SAME LANGUAGE AS THE DOCUMENT: Japanese for Japanese chunks,\n"
    "                English for English chunks. Do NOT put numbers or spec names here.\n\n"
    "ENGLISH EXAMPLE INPUT (chunk from spec Pop-up):\n"
    "  [Source: spec=Pop-up | file=781_(Pop-up)_E_210617.docx]\n"
    "  Table 4-8 Display contents\n"
    "  | Display Contents | Reference        |\n"
    "  | Item Back        | 4.4.1.3          |\n"
    "  | Scroll bar       | SPEC \"Enlarge WA\" |\n\n"
    "ENGLISH EXAMPLE OUTPUT:\n"
    "  [{spec_name: 'Pop-up',     section_number: '4.4.1.3', query: 'display spec of Item Back button'},\n"
    "   {spec_name: 'Enlarge WA', section_number: null,      query: 'display spec of scroll bar'}]\n\n"
    "JAPANESE EXAMPLE INPUT (chunk from Japanese spec Pop-up):\n"
    "  [Source: spec=Pop-up | file=781_(Pop-up)_J_210617.docx]\n"
    "  表4-8 表示内容\n"
    "  | 表示内容       | 参照             |\n"
    "  | アイテムバック | 4.4.1.3          |\n"
    "  | スクロールバー | （EnlargeWA）    |\n\n"
    "JAPANESE EXAMPLE OUTPUT:\n"
    "  [{spec_name: 'Pop-up',     section_number: '4.4.1.3', query: 'アイテムバックボタンの表示仕様'},\n"
    "   {spec_name: 'EnlargeWA',  section_number: null,      query: 'スクロールバーの表示仕様'}]\n\n"
    "Only return references to content NOT already shown in the chunks. "
    "Return an empty list when there are no actionable cross-references."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def should_compress_context(state: dict) -> dict:
    """No-op node — routing logic is in edges.route_after_compression_check."""
    GRAPH_NODE_COUNT.labels(node="should_compress_context").inc()
    return {}


async def detect_cross_references(state: AgentState, llm: BaseChatModel) -> dict:
    """Scan recent RAG tool results for cross-references and extract structured targets.

    Only inspects ToolMessages that arrived after the last cross-reference context
    injection (identified by a HumanMessage starting with '[Cross-reference context]'),
    so repeated passes through the node never re-process the same tool output.

    Uses the regex CrossReferenceDetector as a cheap pre-screen; skips the LLM
    call entirely when no candidate text is found.

    Returns:
        {"cross_ref_targets": [...]} — empty list if nothing detected.
    """
    GRAPH_NODE_COUNT.labels(node="detect_cross_references").inc()
    messages = state.get("messages", [])

    # Find the index of the last cross-ref context injection to avoid reprocessing.
    last_injection_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) and m.content.startswith("[Cross-reference context]"):
            last_injection_idx = i

    recent_tool_msgs = [
        m for m in messages[last_injection_idx + 1:]
        if isinstance(m, ToolMessage) and getattr(m, "name", "") in _RAG_TOOL_NAMES
    ]

    if not recent_tool_msgs:
        return {"cross_ref_targets": []}

    # Fast regex pre-screen — avoids LLM call on clean text.
    detector = CrossReferenceDetector()
    combined = "\n---\n".join(m.content for m in recent_tool_msgs if m.content)
    if not detector.has_references(combined):
        logger.debug("detect_cross_references_skipped", reason="no_refs_in_tool_output")
        return {"cross_ref_targets": []}

    # LLM structured extraction across all recent tool results in one call.
    try:
        structured_llm = llm.with_structured_output(_CrossRefDetectionOutput)
        result: _CrossRefDetectionOutput = await structured_llm.ainvoke([
            HumanMessage(content=f"{_DETECT_SYSTEM}\n\nChunks:\n{combined}"),
        ])
        targets: List[CrossRefTarget] = [
            CrossRefTarget(
                spec_name=t.spec_name,
                section_number=t.section_number or "",
                query=t.query,
            )
            for t in result.targets
            if t.spec_name.strip()
        ]
    except Exception as exc:
        logger.warning("detect_cross_references_llm_failed", error=str(exc))
        return {"cross_ref_targets": []}

    logger.info(
        "cross_references_detected",
        target_count=len(targets),
        tool_msg_count=len(recent_tool_msgs),
    )
    GRAPH_NODE_LATENCY.labels(node="detect_cross_references")
    return {"cross_ref_targets": targets}


async def fetch_cross_ref_context(state: AgentState) -> dict:
    """Fetch content for every CrossRefTarget, inject into messages, clear targets.

    Targets are resolved via CrossReferenceRetriever which applies:
      - Fuzzy spec_name matching (handles typos / case / spacing)
      - 3-tier section fallback (exact → parent prefix → semantic within spec)
      - Semantic-only fallback when spec_name is completely unresolvable

    Fetched contexts are deduplicated against existing message content and
    injected as a single HumanMessage so the orchestrator sees them naturally
    in its next reasoning step.
    """
    GRAPH_NODE_COUNT.labels(node="fetch_cross_ref_context").inc()
    targets = state.get("cross_ref_targets", [])
    if not targets:
        return {"cross_ref_targets": []}

    collection = f"docs_{state['user_id']}"

    try:
        from app.rag.llamaindex.tools import _embed_query

        aclient = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )
        retriever = CrossReferenceRetriever(aclient, embed_fn=_embed_query)
        contexts = await retriever.resolve_targets(list(targets), collection)
    except Exception as exc:
        logger.exception("fetch_cross_ref_context_failed", error=str(exc))
        return {"cross_ref_targets": []}

    if not contexts:
        logger.info("fetch_cross_ref_context_empty", target_count=len(targets))
        return {"cross_ref_targets": []}

    # Deduplicate against all content already in the message history.
    existing_content = {
        m.content.strip()
        for m in state.get("messages", [])
        if hasattr(m, "content") and m.content
    }
    unique = [c for c in contexts if c.strip() not in existing_content]

    if unique:
        injection = "[Cross-reference context]\n" + "\n\n".join(unique)
        logger.info(
            "cross_ref_context_injected",
            context_count=len(unique),
            collection=collection,
        )
        return {
            "messages": [HumanMessage(content=injection)],
            "cross_ref_targets": [],   # clear — consumed
        }

    return {"cross_ref_targets": []}
