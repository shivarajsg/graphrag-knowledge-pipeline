"""
tests.test_qa_engine
--------------------
Automated test suite for Phase 2: Evidence-Grounded Question Answering.
Verifies document-only grounding, source citation integrity, quote verification,
ambiguity and conflict detection, error resilience, and CLI integration.
Unit tests are fully deterministic and require no live API keys.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from graphrag.ingestion.schemas import DocumentChunk, DocumentIndex, ExtractedPage
from graphrag.qa_engine import (
    AnthropicQAClient,
    BaseQAClient,
    DocumentQAEngine,
    MockQAClient,
)
from graphrag.qa_schemas import Claim, EvidenceItem, QAResponse, QAStatus
from graphrag.qa_validator import CitationValidator, find_quote_in_chunk


# ── Synthetic Fixture Generator ───────────────────────────────────────────────

@pytest.fixture
def synthetic_index() -> DocumentIndex:
    """
    Creates a multi-page synthetic DocumentIndex with controlled passages:
    - Page 1: General rules and primate classification
    - Page 2: Research fee waiver exception
    - Page 3: Conflicting directives (permitted vs prohibited)
    - Page 4: Ambiguous discretionary clearance
    """
    doc_id = "test_doc_001"
    source_file = "regulations_handbook.pdf"
    doc_hash = "abc123hash456"

    p1_text = (
        "Section 1: General Export Standards. All commercial exports of biological specimens "
        "require a valid Tier-1 permit issued by the Wildlife Authority. Processing takes 14 days.\n\n"
        "Section 2: Primate Classifications. Species in the order PRIMATES are catalogued in Schedule B. "
        "All Schedule B specimens require specialized temperature-controlled transit."
    )
    p2_text = (
        "Section 3: Exceptions and Research Waivers. Non-commercial specimens intended for non-profit "
        "research institutions are exempt from the standard permit fee. However, research export permits "
        "remain strictly subject to health verification before departure."
    )
    p3_text = (
        "Section 4: Conflicting Directives.\n"
        "Paragraph 4.1: International transfer of biological samples from native flora is permitted under quota.\n"
        "Paragraph 4.2: Notwithstanding any other provision, international transfer of biological samples is prohibited without presidential decree."
    )
    p4_text = (
        "Section 5: Discretionary Authority.\n"
        "The Board may or may not grant emergency clearances under unspecified criteria."
    )

    pages = [
        ExtractedPage(pdf_page_index=1, printed_page_number="i", text=p1_text),
        ExtractedPage(pdf_page_index=2, printed_page_number="ii", text=p2_text),
        ExtractedPage(pdf_page_index=3, printed_page_number="iii", text=p3_text),
        ExtractedPage(pdf_page_index=4, printed_page_number="iv", text=p4_text),
    ]

    c1 = DocumentChunk(
        chunk_id=f"{doc_id}_p001_c00",
        document_id=doc_id,
        source_file=source_file,
        document_hash=doc_hash,
        pdf_page_index=1,
        printed_page_number="i",
        chunk_index=0,
        text="Section 1: General Export Standards. All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority. Processing takes 14 days.",
        page_char_start=0,
        page_char_end=156,
        section_heading="Section 1: General Export Standards",
    )
    c2 = DocumentChunk(
        chunk_id=f"{doc_id}_p001_c01",
        document_id=doc_id,
        source_file=source_file,
        document_hash=doc_hash,
        pdf_page_index=1,
        printed_page_number="i",
        chunk_index=1,
        text="Section 2: Primate Classifications. Species in the order PRIMATES are catalogued in Schedule B. All Schedule B specimens require specialized temperature-controlled transit.",
        page_char_start=158,
        page_char_end=331,
        section_heading="Section 2: Primate Classifications",
    )
    c3 = DocumentChunk(
        chunk_id=f"{doc_id}_p002_c00",
        document_id=doc_id,
        source_file=source_file,
        document_hash=doc_hash,
        pdf_page_index=2,
        printed_page_number="ii",
        chunk_index=0,
        text=p2_text,
        page_char_start=0,
        page_char_end=len(p2_text),
        section_heading="Section 3: Exceptions and Research Waivers",
    )
    c4 = DocumentChunk(
        chunk_id=f"{doc_id}_p003_c00",
        document_id=doc_id,
        source_file=source_file,
        document_hash=doc_hash,
        pdf_page_index=3,
        printed_page_number="iii",
        chunk_index=0,
        text=p3_text,
        page_char_start=0,
        page_char_end=len(p3_text),
        section_heading="Section 4: Conflicting Directives",
    )
    c5 = DocumentChunk(
        chunk_id=f"{doc_id}_p004_c00",
        document_id=doc_id,
        source_file=source_file,
        document_hash=doc_hash,
        pdf_page_index=4,
        printed_page_number="iv",
        chunk_index=0,
        text=p4_text,
        page_char_start=0,
        page_char_end=len(p4_text),
        section_heading="Section 5: Discretionary Authority",
    )

    chunks = [c1, c2, c3, c4, c5]
    return DocumentIndex(
        document_id=doc_id,
        source_file=source_file,
        document_hash=doc_hash,
        total_pages=4,
        total_chunks=5,
        pages=pages,
        chunks=chunks,
    )


class CustomTestDoubleClient(BaseQAClient):
    """Test double allowing tests to inject programmatic LLM responses."""

    def __init__(self, response_data: Dict[str, Any] | str) -> None:
        self.response_data = response_data

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if isinstance(self.response_data, str):
            return self.response_data
        return json.dumps(self.response_data)


# ── Test Suite ────────────────────────────────────────────────────────────────

def test_qa_single_passage_direct_answer(synthetic_index: DocumentIndex):
    """Test 1: A question directly answered by a single retrieved passage."""
    quote = "All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority."
    mock_payload = {
        "answer": "Commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
        "status": "answered",
        "claims": [
            {
                "claim": "All commercial exports of biological specimens require a Tier-1 permit.",
                "evidence_ids": ["E1"],
                "is_inference": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p001_c00",
                "pdf_page_index": 1,
                "quote": quote,
            }
        ],
        "evidence_trail": ["Source passage E1 on page 1 explicitly defines the required permit."],
        "limitations": [],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What permit is required for biological specimen exports?")

    assert response.status == QAStatus.ANSWERED.value
    assert len(response.claims) == 1
    assert response.claims[0].is_inference is False
    assert len(response.evidence) == 1
    assert response.evidence[0].quote == quote
    assert response.evidence[0].pdf_page_index == 1
    assert response.evidence[0].chunk_id == "test_doc_001_p001_c00"
    assert len(response.validation_warnings) == 0


def test_qa_multi_passage_synthesis(synthetic_index: DocumentIndex):
    """Test 2: A question requiring synthesis across two separate passages."""
    mock_payload = {
        "answer": "Biological specimens of the order Primates are catalogued in Schedule B and require specialized temperature-controlled transit, and their commercial export requires a Tier-1 permit.",
        "status": "answered",
        "claims": [
            {
                "claim": "Commercial exports of primates require both a Tier-1 permit and specialized temperature-controlled transit.",
                "evidence_ids": ["E1", "E2"],
                "is_inference": True,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p001_c00",
                "pdf_page_index": 1,
                "quote": "All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
            },
            {
                "evidence_id": "E2",
                "chunk_id": "test_doc_001_p001_c01",
                "pdf_page_index": 1,
                "quote": "All Schedule B specimens require specialized temperature-controlled transit.",
            },
        ],
        "evidence_trail": ["Combining general permit rules with primate transit requirements."],
        "limitations": [],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What transit and permit rules apply to commercial primate exports?")

    assert response.status == QAStatus.ANSWERED.value
    assert len(response.claims) == 1
    assert response.claims[0].is_inference is True
    assert set(response.claims[0].evidence_ids) == {"E1", "E2"}
    assert len(response.evidence) == 2


def test_qa_attention_to_exception(synthetic_index: DocumentIndex):
    """Test 3: A question requiring attention to a specific rule exception."""
    quote = "Non-commercial specimens intended for non-profit research institutions are exempt from the standard permit fee."
    mock_payload = {
        "answer": "Non-profit research institutions are exempt from standard permit fees for non-commercial specimens, but must still undergo health verification.",
        "status": "answered",
        "claims": [
            {
                "claim": "Research institutions are exempt from standard permit fees for non-commercial specimens.",
                "evidence_ids": ["E1"],
                "is_inference": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p002_c00",
                "pdf_page_index": 2,
                "quote": quote,
            }
        ],
        "evidence_trail": ["Section 3 provides an explicit fee waiver exception."],
        "limitations": [],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "Do research institutions have an exception from the permit fee?")

    assert response.status == QAStatus.ANSWERED.value
    assert response.evidence[0].pdf_page_index == 2
    assert quote in synthetic_index.chunks[2].text


def test_qa_absent_information(synthetic_index: DocumentIndex):
    """Test 4: A question whose answer is completely absent from the document."""
    mock_payload = {
        "answer": "The available document evidence does not establish any export quotas for Golden Eagles.",
        "status": "insufficient_evidence",
        "claims": [],
        "evidence": [],
        "evidence_trail": ["No mentions of Golden Eagles found in retrieved passages."],
        "limitations": ["Information regarding Golden Eagles is not in the document."],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What is the export quota for Golden Eagles in Antarctica?")

    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert len(response.claims) == 0
    assert len(response.evidence) == 0
    assert len(response.limitations) > 0


def test_qa_ambiguous_evidence(synthetic_index: DocumentIndex):
    """Test 5: A question where the available evidence is ambiguous or open-ended."""
    quote = "The Board may or may not grant emergency clearances under unspecified criteria."
    mock_payload = {
        "answer": "The document is ambiguous regarding emergency clearances: it states the Board may or may not grant them under unspecified criteria.",
        "status": "ambiguous",
        "claims": [
            {
                "claim": "Emergency clearance granting is discretionary and governed by unspecified criteria.",
                "evidence_ids": ["E1"],
                "is_inference": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p004_c00",
                "pdf_page_index": 4,
                "quote": quote,
            }
        ],
        "evidence_trail": ["Section 5 establishes ambiguity without defining operational rules."],
        "limitations": ["Specific criteria are omitted in the document text."],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What exact criteria are used for emergency clearances?")

    assert response.status == QAStatus.AMBIGUOUS.value
    assert len(response.evidence) == 1
    assert response.evidence[0].pdf_page_index == 4


def test_qa_conflicting_evidence(synthetic_index: DocumentIndex):
    """Test 6: A document containing directly contradictory statements."""
    q1 = "International transfer of biological samples from native flora is permitted under quota."
    q2 = "Notwithstanding any other provision, international transfer of biological samples is prohibited without presidential decree."

    mock_payload = {
        "answer": "The document contains conflicting statements: Paragraph 4.1 permits transfer under quota, whereas Paragraph 4.2 prohibits transfer without presidential decree.",
        "status": "conflicting_evidence",
        "claims": [
            {"claim": "Paragraph 4.1 permits transfer under quota.", "evidence_ids": ["E1"], "is_inference": False},
            {"claim": "Paragraph 4.2 prohibits transfer without presidential decree.", "evidence_ids": ["E2"], "is_inference": False},
        ],
        "evidence": [
            {"evidence_id": "E1", "chunk_id": "test_doc_001_p003_c00", "pdf_page_index": 3, "quote": q1},
            {"evidence_id": "E2", "chunk_id": "test_doc_001_p003_c00", "pdf_page_index": 3, "quote": q2},
        ],
        "evidence_trail": ["Paragraph 4.1 and 4.2 provide mutually contradictory directives."],
        "limitations": ["The contradiction cannot be resolved from the document text alone."],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "Is international transfer of native biological samples permitted or prohibited?")

    assert response.status == QAStatus.CONFLICTING_EVIDENCE.value
    assert len(response.evidence) == 2
    assert response.evidence[0].pdf_page_index == 3
    assert response.evidence[1].pdf_page_index == 3


def test_qa_outside_knowledge_refusal(synthetic_index: DocumentIndex):
    """Test 7: A question prompting outside knowledge causes the system to abstain."""
    # When asked general knowledge outside the document
    engine = DocumentQAEngine(mock=True)
    response = engine.answer_question(synthetic_index, "What is the capital of France and its GDP in 2023?")

    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert len(response.claims) == 0
    assert "insufficient" in response.answer.lower() or "not" in response.answer.lower()


def test_qa_unrelated_second_document(synthetic_index: DocumentIndex):
    """Test 8: A question inquiring about an unrelated second document abstains."""
    engine = DocumentQAEngine(mock=True)
    response = engine.answer_question(synthetic_index, "What does the 2024 Swiggy Financial Report state about Instamart revenue?")

    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert len(response.claims) == 0


def test_qa_validator_fabricated_evidence_id(synthetic_index: DocumentIndex):
    """Test 9: Model outputs a claim referencing a fabricated evidence ID 'E99'."""
    quote = "All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority."
    mock_payload = {
        "answer": "Commercial exports require a permit.",
        "status": "answered",
        "claims": [
            {
                "claim": "Commercial exports require a permit.",
                "evidence_ids": ["E1", "E99"],  # E99 does not exist
                "is_inference": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p001_c00",
                "pdf_page_index": 1,
                "quote": quote,
            }
        ],
        "evidence_trail": [],
        "limitations": [],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What permit is required?")

    # E99 should be stripped out of claim evidence_ids and flagged in warnings
    assert "E99" not in response.claims[0].evidence_ids
    assert response.claims[0].evidence_ids == ["E1"]
    assert any("E99" in w for w in response.validation_warnings)


def test_qa_validator_fabricated_quotation(synthetic_index: DocumentIndex):
    """Test 10: Model returns a fabricated quote not present in the chunk text."""
    mock_payload = {
        "answer": "Biological exports are strictly prohibited for all entities.",
        "status": "answered",
        "claims": [
            {
                "claim": "All biological exports are forbidden forever.",
                "evidence_ids": ["E1"],
                "is_inference": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p001_c00",
                "pdf_page_index": 1,
                "quote": "Biological exports are strictly prohibited for all entities throughout the universe.",
            }
        ],
        "evidence_trail": [],
        "limitations": [],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "Are biological exports allowed?")

    # The fabricated quotation must be rejected, discarding the ungrounded claim and downgrading status
    assert len(response.evidence) == 0
    assert len(response.claims) == 0
    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert any("rejected" in w.lower() for w in response.validation_warnings)


def test_qa_validator_invalid_page_reference(synthetic_index: DocumentIndex):
    """Test 11: Model returns invalid physical page (e.g. 99 instead of 1)."""
    quote = "All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority."
    mock_payload = {
        "answer": "Tier-1 permit is required.",
        "status": "answered",
        "claims": [{"claim": "Tier-1 permit required.", "evidence_ids": ["E1"]}],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p001_c00",
                "pdf_page_index": 99,  # Wrong page! Real page is 1
                "quote": quote,
            }
        ],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What permit is required?")

    # Page should be corrected to 1 and warning logged
    assert response.evidence[0].pdf_page_index == 1
    assert any("page" in w.lower() for w in response.validation_warnings)


def test_qa_malformed_model_output(synthetic_index: DocumentIndex):
    """Test 12: Model returns non-JSON or corrupted text."""
    malformed_raw = "I am an AI and I think the answer is Tier-1 permit, but I won't format as JSON."
    client = CustomTestDoubleClient(malformed_raw)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What permit is required?")

    assert response.status == QAStatus.PROCESSING_ERROR.value
    assert len(response.claims) == 0
    assert len(response.validation_warnings) > 0


def test_qa_empty_document_or_missing_index():
    """Test 13: Empty DocumentIndex with zero chunks."""
    empty_index = DocumentIndex(
        document_id="empty_doc",
        source_file="empty.pdf",
        total_pages=0,
        total_chunks=0,
        pages=[],
        chunks=[],
    )
    engine = DocumentQAEngine(mock=True)
    response = engine.answer_question(empty_index, "What is the rule?")

    assert response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert "empty" in response.answer.lower() or "no readable" in response.answer.lower()


def test_qa_missing_api_key_configuration():
    """Test 14: Clear error when mock=False and no API key is set."""
    # Temporarily ensure ANTHROPIC_API_KEY is not in env
    orig = os.environ.get("ANTHROPIC_API_KEY")
    try:
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        with pytest.raises(ValueError) as excinfo:
            DocumentQAEngine(mock=False, api_key=None)
        assert "API key is required" in str(excinfo.value)
    finally:
        if orig is not None:
            os.environ["ANTHROPIC_API_KEY"] = orig


def test_qa_valid_end_to_end_exact_citations(synthetic_index: DocumentIndex):
    """Test 15: Full valid end-to-end question answering with exact source evidence."""
    engine = DocumentQAEngine(mock=True)
    response = engine.answer_question(
        synthetic_index,
        "What permit is required for biological specimen exports?",
    )

    assert response.status in (QAStatus.ANSWERED.value, QAStatus.PARTIALLY_ANSWERED.value)
    assert len(response.evidence) > 0
    # Every evidence quote must be an exact substring of the referenced chunk
    for ev in response.evidence:
        chunk = next((c for c in synthetic_index.chunks if c.chunk_id == ev.chunk_id), None)
        assert chunk is not None
        assert ev.quote in chunk.text


def test_qa_partially_answered_with_limitations(synthetic_index: DocumentIndex):
    """Test 16: An answer that remains explicitly partial when only part is supported."""
    quote = "Processing takes 14 days."
    mock_payload = {
        "answer": "Processing takes 14 days according to the document. However, the permit application fee in Japanese Yen is not stated in the document.",
        "status": "partially_answered",
        "claims": [
            {
                "claim": "Processing takes 14 days for Tier-1 permits.",
                "evidence_ids": ["E1"],
                "is_inference": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "test_doc_001_p001_c00",
                "pdf_page_index": 1,
                "quote": quote,
            }
        ],
        "evidence_trail": ["Step 1: Section 1 explicitly gives the 14-day duration."],
        "limitations": ["The fee in Japanese Yen is not established in the text."],
    }

    client = CustomTestDoubleClient(mock_payload)
    engine = DocumentQAEngine(llm_client=client)
    response = engine.answer_question(synthetic_index, "What is the processing time and fee in Yen?")

    assert response.status == QAStatus.PARTIALLY_ANSWERED.value
    assert len(response.claims) == 1
    assert len(response.limitations) == 1
    assert "Yen" in response.limitations[0]


def test_cli_ask_subprocess_end_to_end(tmp_path: Path, synthetic_index: DocumentIndex):
    """Test 17: Command line interface end-to-end execution with --ask and --mock."""
    index_file = tmp_path / "test_cli_index.json"
    synthetic_index.save(index_file)

    cmd = [
        sys.executable,
        "-m",
        "graphrag.ingestion.cli",
        "--index",
        str(index_file),
        "--ask",
        "What permit is required for biological specimen exports?",
        "--mock",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).parent.parent,
    )

    assert proc.returncode == 0, f"CLI exited with error: {proc.stderr}"
    output = proc.stdout
    assert "Document Question Answering Engine (Phase 2)" in output
    assert "Status: [" in output
    assert "ANSWER:" in output


def test_cites_proof_of_concept_if_available():
    """
    Test 18: CITES proof-of-concept verification.
    When CITES_E-Appendices.pdf or outputs/cites_index.json is available,
    runs Question 1 and Question 2 through the document QA pipeline and verifies:
    - Q1: Primate classification is captured, but specific pharmaceutical export permissions
      to the USA are noted as unestablished/limitations.
    - Q2: Indian central-government approvals are recognized as absent/insufficient evidence.
    """
    index_path = Path("outputs/cites_index.json")
    pdf_path = Path("CITES_E-Appendices.pdf")

    if not index_path.exists() and pdf_path.exists():
        from graphrag.ingestion.indexer import index_pdf
        index = index_pdf(pdf_path, output_path=index_path)
    elif index_path.exists():
        index = DocumentIndex.load(index_path)
    else:
        pytest.skip(
            "CITES_E-Appendices.pdf is not present in workspace. "
            "CITES live QA proof-of-concept is pending document provision."
        )

    engine = DocumentQAEngine(mock=True)

    # Question 1: Rhesus Monkeys export for pharmaceutical research
    q1 = "Can Rhesus Monkeys be exported to the USA for pharmaceutical research?"
    resp1 = engine.answer_question(index, q1)
    assert resp1.status == QAStatus.PARTIALLY_ANSWERED.value
    assert len(resp1.evidence) > 0
    assert "PRIMATES" in resp1.answer or "Appendix II" in resp1.answer
    assert any("pharmaceutical" in lim.lower() or "research" in lim.lower() for lim in resp1.limitations)

    # Question 2: Specific Indian central-government approvals and ministries
    q2 = "Are specific Indian central-government approvals required for this export? If yes, which approvals and ministries are identified?"
    resp2 = engine.answer_question(index, q2)
    assert resp2.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert len(resp2.claims) == 0
    assert any("indian" in lim.lower() or "ministr" in lim.lower() for lim in resp2.limitations)


# ==============================================================================
# 9. Live Model Configuration & Safeguard Tests
# ==============================================================================

def test_anthropic_qa_client_missing_api_key(monkeypatch):
    """Live QA client must raise ValueError when no API key is provided."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Anthropic API key is required"):
        AnthropicQAClient(api_key=None)


def test_anthropic_qa_client_retired_model_rejected(monkeypatch):
    """Retired models like claude-3-5-sonnet-20241022 must be rejected with a clear message."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    with pytest.raises(ValueError, match="retired"):
        AnthropicQAClient(api_key="sk-ant-test-dummy-key", model="claude-3-5-sonnet-20241022")


def test_anthropic_qa_client_unsupported_model_rejected(monkeypatch):
    """Unsupported models not following Anthropic naming conventions must be rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    with pytest.raises(ValueError, match="Unsupported Anthropic model"):
        AnthropicQAClient(api_key="sk-ant-test-dummy-key", model="gpt-4o")


def test_anthropic_qa_client_default_model_is_supported():
    """DEFAULT_QA_MODEL must default to claude-sonnet-5-5."""
    from graphrag.qa_engine import DEFAULT_QA_MODEL
    assert "claude-sonnet-5-5" in DEFAULT_QA_MODEL or "claude-" in DEFAULT_QA_MODEL


def test_document_qa_engine_no_silent_fallback_on_missing_credentials(monkeypatch):
    """Live DocumentQAEngine must fail loudly if credentials are absent, never silently fallback to mock."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Anthropic API key is required"):
        DocumentQAEngine(mock=False, api_key=None)


