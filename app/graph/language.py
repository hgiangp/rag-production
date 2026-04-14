"""Shared language helpers for graph nodes and API layer.

Centralises the BCP-47 code → human-readable name mapping and the
language-instruction string that is appended to LLM system prompts.
Import from here instead of duplicating in individual modules.
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
    """Return the human-readable language name for a BCP-47 code.

    Falls back to the code itself when not found (e.g. "zh-TW" → "zh-TW").
    """
    return _LANGUAGE_NAMES.get(code.lower(), code)


def lang_instruction(target_language: str) -> str:
    """Return a language-enforcement suffix to append to any system prompt.

    Intentionally omitted from compress_context — that node summarises
    retrieved context verbatim and must not translate mid-pipeline.
    """
    name = lang_name(target_language)
    return (
        f"\n\nLANGUAGE REQUIREMENT: You MUST write your entire response in {name}. "
        f"Retrieved context may be in a different language — read and understand it, "
        f"then produce your answer entirely in {name}. "
        f"Do NOT mix languages or leave untranslated fragments."
    )
