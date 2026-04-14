"""Shared language utilities for graph nodes.

Centralises BCP-47 → human-readable name mapping and prompt helpers so that
query.py and generation.py stay in sync without duplication.
"""

from typing import Dict

_LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ja": "Japanese",
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "pt": "Portuguese",
    "th": "Thai",
    "ar": "Arabic",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "id": "Indonesian",
}


def lang_name(code: str) -> str:
    """Return a human-readable language name for a BCP-47 code."""
    return _LANGUAGE_NAMES.get(code.lower(), code)


def lang_instruction(target_language: str) -> str:
    """Return a system-prompt suffix that forces the LLM to answer in *target_language*.

    Appended to orchestrator / fallback / aggregate system prompts.
    Intentionally NOT used in compress_context — that node summarises retrieved
    source material and must preserve factual fidelity over language consistency.
    """
    name = lang_name(target_language)
    return (
        f"\n\nLANGUAGE REQUIREMENT: You MUST write your entire response in {name}. "
        f"Retrieved context may be in a different language — read and understand it, "
        f"then produce your answer entirely in {name}. "
        f"Do NOT mix languages or leave untranslated fragments."
    )
