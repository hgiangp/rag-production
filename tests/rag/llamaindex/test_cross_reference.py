"""Tests for cross-reference detection and metadata extraction."""

import pytest

from app.rag.llamaindex.cross_reference import (
    CrossReferenceDetector,
    DocumentMetadata,
    extract_section_number,
)


# ── DocumentMetadata.from_filename ─────────────────────────────────────────────

class TestDocumentMetadata:
    def test_parses_standard_filename(self):
        meta = DocumentMetadata.from_filename("7821ZXXXXQ000_E_(Pop-up)_210617.pdf")
        assert meta is not None
        assert meta.model_symbol == "7821ZXXXXQ000"
        assert meta.spec_name == "Pop-up"
        assert meta.language == "E"
        assert meta.version == "210617"
        assert meta.raw_filename == "7821ZXXXXQ000_E_(Pop-up)_210617.pdf"

    def test_parses_japanese_version(self):
        meta = DocumentMetadata.from_filename("7821ZXXXXQ000_J_(Pop-up)_210617.pdf")
        assert meta is not None
        assert meta.language == "J"

    def test_parses_without_parentheses(self):
        meta = DocumentMetadata.from_filename("7821ZXXXXQ000_E_Pop-up_210617.pdf")
        assert meta is not None
        assert meta.spec_name == "Pop-up"

    def test_returns_none_for_unknown_format(self):
        meta = DocumentMetadata.from_filename("random_document.docx")
        assert meta.spec_name is None

    def test_handles_no_extension(self):
        meta = DocumentMetadata.from_filename("7821ZXXXXQ000_E_Pop-up_210617")
        assert meta is not None
        assert meta.spec_name == "Pop-up"


# ── extract_section_number ──────────────────────────────────────────────────────

class TestExtractSectionNumber:
    @pytest.mark.parametrize("heading,expected", [
        ("3.1.2 Overview", "3.1.2"),
        ("3.1.2.3 Details", "3.1.2.3"),
        ("1 Introduction", "1"),
        ("Section 4.5 Appendix", "4.5"),
        ("No section here", None),
        ("", None),
    ])
    def test_extracts_section(self, heading, expected):
        assert extract_section_number(heading) == expected


# ── CrossReferenceDetector ──────────────────────────────────────────────────────

class TestCrossReferenceDetector:
    def setup_method(self):
        self.detector = CrossReferenceDetector()

    # Section references
    def test_detects_see_section(self):
        refs = self.detector.detect("For details, see section 3.1.2.3 below.")
        assert len(refs.section_refs) == 1
        assert refs.section_refs[0].section_number == "3.1.2.3"

    def test_detects_refer_to_section(self):
        refs = self.detector.detect("Refer to section 4.2 for configuration.")
        assert len(refs.section_refs) == 1
        assert refs.section_refs[0].section_number == "4.2"

    def test_detects_japanese_section(self):
        refs = self.detector.detect("詳細は3.1.2.3項を参照。")
        assert any(r.section_number == "3.1.2.3" for r in refs.section_refs)

    def test_deduplicates_section_refs(self):
        refs = self.detector.detect("See section 3.1 and also refer to section 3.1.")
        section_nums = [r.section_number for r in refs.section_refs]
        assert section_nums.count("3.1") == 1

    # Document references
    def test_detects_spec_in_quotes(self):
        refs = self.detector.detect("As defined in spec 'EnlargeWA'.")
        assert any(r.spec_name == "EnlargeWA" for r in refs.document_refs)

    def test_detects_parenthesized_spec(self):
        refs = self.detector.detect("This is covered by (EnlargeWA) specification.")
        assert any(r.spec_name == "EnlargeWA" for r in refs.document_refs)

    def test_detects_double_quoted_spec(self):
        refs = self.detector.detect('See specification "EnlargeWA" for details.')
        assert any(r.spec_name == "EnlargeWA" for r in refs.document_refs)

    def test_no_false_positive_common_words(self):
        refs = self.detector.detect("See the document for details.")
        assert all(r.spec_name.lower() not in {"the", "this", "see"} for r in refs.document_refs)

    def test_detects_both_types(self):
        text = "See section 3.1.2 and refer to spec 'EnlargeWA' for full context."
        refs = self.detector.detect(text)
        assert len(refs.section_refs) >= 1
        assert len(refs.document_refs) >= 1

    def test_empty_text(self):
        refs = self.detector.detect("")
        assert refs.section_refs == []
        assert refs.document_refs == []

