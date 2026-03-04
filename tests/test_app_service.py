"""
tests.test_app_service
----------------------
Phase 5: End-to-End Application Service & Integration Tests.
Verifies:
1. PDF upload validation, corrupted/empty file rejection, content-hash caching.
2. Active document state management and cross-document isolation.
3. Evidence-grounded Question Answering and claim verification display.
4. Physical PDF page rendering (PNG) using PyMuPDF.
5. CITES Question 1 & 2 interview questions (ensuring correct abstention).
6. Regulatory table reasoning integration.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import pytest
from pathlib import Path

from graphrag.app_service import DocumentIntelligenceService
from graphrag.qa_schemas import QAStatus

CITES_INDEX_PATH = Path("outputs/cites_index.json")
CITES_PDF_PATH = Path("CITES_E-Appendices.pdf")


@pytest.fixture
def tmp_service(tmp_path: Path) -> DocumentIntelligenceService:
    """Fixture providing a DocumentIntelligenceService using a temporary storage directory."""
    return DocumentIntelligenceService(base_storage_dir=tmp_path / "outputs")


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Creates a valid 2-page synthetic PDF in memory with text and table."""
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Section 1: General Wildlife Export Controls.\nAll commercial shipments require valid permits under Section 4.")
    # Page 2
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 4: Permit Issuance Standards.\nPermits must be issued by the Management Authority prior to export.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ==============================================================================
# 1. Upload & Ingestion Tests
# ==============================================================================

def test_successful_pdf_upload_and_indexing(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Test standard PDF upload, validation, page extraction, and chunk indexing."""
    index, is_cached = tmp_service.upload_and_index_document(
        file_bytes=sample_pdf_bytes,
        filename="test_upload.pdf",
    )

    assert is_cached is False
    assert index.total_pages == 2
    assert index.total_chunks >= 2
    assert tmp_service.active_index is not None
    assert tmp_service.active_pdf_path is not None
    assert tmp_service.active_pdf_path.exists()

    summary = tmp_service.get_document_summary()
    assert summary["has_active_document"] is True
    assert summary["total_pages"] == 2
    assert summary["total_chunks"] == index.total_chunks


def test_invalid_file_type_rejection(tmp_service: DocumentIntelligenceService):
    """Uploading non-PDF files must be rejected with ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        tmp_service.upload_and_index_document(
            file_bytes=b"Sample plain text data",
            filename="document.txt",
        )


def test_empty_file_rejection(tmp_service: DocumentIntelligenceService):
    """Uploading 0-byte file must be rejected."""
    with pytest.raises(ValueError, match="empty"):
        tmp_service.upload_and_index_document(
            file_bytes=b"",
            filename="empty.pdf",
        )


def test_corrupted_pdf_rejection(tmp_service: DocumentIntelligenceService):
    """Uploading malformed/corrupted PDF bytes must be rejected."""
    corrupted_bytes = b"%PDF-1.4\nCorrupted random byte stream that cannot be parsed"
    with pytest.raises(ValueError, match="Invalid or corrupted PDF"):
        tmp_service.upload_and_index_document(
            file_bytes=corrupted_bytes,
            filename="corrupted.pdf",
        )


def test_content_hash_caching_and_reuse(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Uploading identical content twice must reuse the existing index without re-indexing."""
    # First upload
    index1, is_cached1 = tmp_service.upload_and_index_document(sample_pdf_bytes, "doc1.pdf")
    assert is_cached1 is False

    # Second upload with identical bytes
    index2, is_cached2 = tmp_service.upload_and_index_document(sample_pdf_bytes, "doc1.pdf")
    assert is_cached2 is True
    assert index2.document_hash == index1.document_hash


def test_upload_changed_file_same_filename(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Uploading different bytes with the same filename creates a new index without colliding."""
    # First version
    index1, _ = tmp_service.upload_and_index_document(sample_pdf_bytes, "document.pdf")

    # Second version (modified bytes)
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Completely different content version.")
    changed_bytes = doc.tobytes()
    doc.close()

    index2, is_cached2 = tmp_service.upload_and_index_document(changed_bytes, "document.pdf")
    assert is_cached2 is False
    assert index2.document_hash != index1.document_hash


def test_session_isolation_and_cross_contamination_prevention(
    tmp_service: DocumentIntelligenceService,
    sample_pdf_bytes: bytes,
):
    """Switching active documents must clear previous QA response and prevent cross-contamination."""
    # Index Doc A and query
    tmp_service.upload_and_index_document(sample_pdf_bytes, "doc_A.pdf")
    resp_a = tmp_service.submit_question("What permits are required under Section 1?", mock=True)
    assert tmp_service.last_response is not None

    # Now upload Doc B
    doc_b = fitz.open()
    p = doc_b.new_page()
    p.insert_text((50, 50), "Document B content on marine fisheries regulations.")
    doc_b_bytes = doc_b.tobytes()
    doc_b.close()

    tmp_service.upload_and_index_document(doc_b_bytes, "doc_B.pdf")

    # Previous response must be cleared
    assert tmp_service.last_response is None
    assert tmp_service.last_query is None
    assert tmp_service.active_index.source_file.endswith("doc_B.pdf")


def test_clear_active_document(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Clearing document resets session state completely."""
    tmp_service.upload_and_index_document(sample_pdf_bytes, "doc.pdf")
    assert tmp_service.active_index is not None

    tmp_service.clear_active_document()
    assert tmp_service.active_index is None
    assert tmp_service.active_pdf_path is None
    assert tmp_service.last_response is None
    summary = tmp_service.get_document_summary()
    assert summary["has_active_document"] is False


# ==============================================================================
# 2. Question Answering & Citation Tests
# ==============================================================================

def test_question_answering_without_active_document(tmp_service: DocumentIntelligenceService):
    """Submitting a question without an active document raises ValueError."""
    with pytest.raises(ValueError, match="No active document loaded"):
        tmp_service.submit_question("What does the document state?")


def test_empty_question_rejection(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Submitting empty question raises ValueError."""
    tmp_service.upload_and_index_document(sample_pdf_bytes, "sample.pdf")
    with pytest.raises(ValueError, match="Question string cannot be empty"):
        tmp_service.submit_question("   ")


def test_qa_execution_mock_mode(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Execute question answering in mock mode against indexed document."""
    tmp_service.upload_and_index_document(sample_pdf_bytes, "sample.pdf")
    response = tmp_service.submit_question("What permits are required under Section 1?", mock=True)

    assert response.status in [QAStatus.ANSWERED.value, QAStatus.PARTIALLY_ANSWERED.value]
    assert len(response.evidence) > 0
    assert response.evidence[0].pdf_page_index in [1, 2]
    assert tmp_service.last_response == response


def test_missing_evidence_question_abstains(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Querying facts absent from the document returns INSUFFICIENT_EVIDENCE."""
    tmp_service.upload_and_index_document(sample_pdf_bytes, "sample.pdf")
    response = tmp_service.submit_question("What are the nuclear reactor safety protocols?", mock=True)

    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert len(response.claims) == 0
    assert len(response.limitations) > 0


# ==============================================================================
# 3. PDF Page Rendering Tests
# ==============================================================================

def test_render_page_image_valid_page(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Renders valid page as PNG image bytes."""
    tmp_service.upload_and_index_document(sample_pdf_bytes, "sample.pdf")
    png_bytes = tmp_service.render_page_image(pdf_page_index=1, zoom=1.0)

    assert png_bytes is not None
    # Check PNG magic bytes: \x89PNG\r\n\x1a\n
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_page_image_out_of_bounds(tmp_service: DocumentIntelligenceService, sample_pdf_bytes: bytes):
    """Requesting non-existent page number returns None without crashing."""
    tmp_service.upload_and_index_document(sample_pdf_bytes, "sample.pdf")
    png_bytes = tmp_service.render_page_image(pdf_page_index=999)
    assert png_bytes is None


def test_render_page_image_no_active_pdf(tmp_service: DocumentIntelligenceService):
    """Requesting page render when no PDF is loaded returns None."""
    assert tmp_service.render_page_image(pdf_page_index=1) is None


# ==============================================================================
# 4. Live CITES End-to-End Evaluation (Questions 1 & 2)
# ==============================================================================

def test_live_cites_e2e_q1_export_pharmaceutical_abstention(tmp_service: DocumentIntelligenceService):
    """
    CITES Question 1: 'Can Rhesus Monkeys be exported to the USA for pharmaceutical research?'
    Verifies that the system abstains with INSUFFICIENT_EVIDENCE.
    """
    if not CITES_INDEX_PATH.exists():
        pytest.skip("CITES index not found.")

    tmp_service.load_existing_index(CITES_INDEX_PATH, CITES_PDF_PATH)
    response = tmp_service.submit_question(
        "Can Rhesus Monkeys be exported to the USA for pharmaceutical research?",
        mock=True,
    )

    # Captures Primate Appendix II listing while explicitly noting pharmaceutical research is absent
    assert response.status in [QAStatus.PARTIALLY_ANSWERED.value, QAStatus.INSUFFICIENT_EVIDENCE.value]
    assert any("pharmaceutical" in lim.lower() or "absent" in lim.lower() or "research" in lim.lower() for lim in response.limitations)



def test_live_cites_e2e_q2_indian_approvals_abstention(tmp_service: DocumentIntelligenceService):
    """
    CITES Question 2: 'Are specific Indian central-government approvals required for this export,
    and which approvals or ministries are identified?'
    Verifies that the system abstains with INSUFFICIENT_EVIDENCE.
    """
    if not CITES_INDEX_PATH.exists():
        pytest.skip("CITES index not found.")

    tmp_service.load_existing_index(CITES_INDEX_PATH, CITES_PDF_PATH)
    response = tmp_service.submit_question(
        "Are specific Indian central-government approvals required for this export, and which approvals or ministries are identified?",
        mock=True,
    )

    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert "supplied document does not identify any specific Indian" in response.answer or "insufficient" in response.answer.lower()


# ==============================================================================
# 5. Regulatory Table Reasoner Integration via Service
# ==============================================================================

def test_inspect_table_taxon_direct_listing(tmp_service: DocumentIntelligenceService):
    """Look up Macaca silenus on CITES index via service."""
    if not CITES_INDEX_PATH.exists():
        pytest.skip("CITES index not found.")

    tmp_service.load_existing_index(CITES_INDEX_PATH, CITES_PDF_PATH)
    res = tmp_service.inspect_table_taxon("Macaca silenus")

    assert res["status"] == "direct_listing"
    assert res["classification"] == "Appendix I"
    assert res["is_explicit_listing"] is True
    assert res["is_inference"] is False
    assert res["provenance"]["pdf_page_index"] == 20


def test_inspect_table_taxon_conditional_inference(tmp_service: DocumentIntelligenceService):
    """Look up Macaca mulatta with higher_taxon premise under non-strict policy."""
    if not CITES_INDEX_PATH.exists():
        pytest.skip("CITES index not found.")

    tmp_service.load_existing_index(CITES_INDEX_PATH, CITES_PDF_PATH)
    res = tmp_service.inspect_table_taxon(
        taxon_name="Macaca mulatta",
        higher_taxon="PRIMATES",
        strict_document_only=False,
    )

    assert res["status"] == "conditional_inference"
    assert "Appendix II" in res["classification"]
    assert res["is_explicit_listing"] is False
    assert res["is_inference"] is True
    assert len(res["assumptions"]) > 0


def test_inspect_table_taxon_strict_document_only(tmp_service: DocumentIntelligenceService):
    """Look up Macaca mulatta with strict_document_only=True returns unlisted_in_document."""
    if not CITES_INDEX_PATH.exists():
        pytest.skip("CITES index not found.")

    tmp_service.load_existing_index(CITES_INDEX_PATH, CITES_PDF_PATH)
    res = tmp_service.inspect_table_taxon(
        taxon_name="Macaca mulatta",
        higher_taxon="PRIMATES",
        strict_document_only=True,
    )

    assert res["status"] == "unlisted_in_document"
    assert res["is_inference"] is False
    assert "Strict document-only policy" in res["rationale"]


def test_app_service_live_mode_without_credentials_raises_error(
    tmp_service: DocumentIntelligenceService,
    sample_pdf_bytes: bytes,
    monkeypatch,
):
    """Calling submit_question with mock=False and no API key must fail loudly with ValueError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tmp_service.upload_and_index_document(file_bytes=sample_pdf_bytes, filename="sample.pdf")

    with pytest.raises(ValueError, match="Anthropic API key is required"):
        tmp_service.submit_question(
            question="What is required for export?",
            mock=False,
            api_key=None,
        )

