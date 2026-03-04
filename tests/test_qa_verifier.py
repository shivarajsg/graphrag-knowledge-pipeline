"""
tests.test_qa_verifier
----------------------
Comprehensive automated test suite for Phase 3: Claim-Level Evidence Verification
and Hallucination Safeguards.
Covers:
  - Layer A: Citation Integrity & Provenance Validation
  - Layer B: Proposition Completeness & Exception Detection
  - Layer C: Semantic Claim-to-Evidence Verification
  - End-to-End QA Engine integration, abstention, contradiction resolution, and CLI.
All tests are deterministic and execute without requiring live API keys.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from graphrag.ingestion.schemas import DocumentChunk, DocumentIndex, ExtractedPage
from graphrag.qa_engine import DocumentQAEngine, MockQAClient
from graphrag.qa_schemas import (
    Claim,
    ClaimVerificationStatus,
    EvidenceItem,
    QAResponse,
    QAStatus,
)
from graphrag.qa_verifier import (
    BaseSemanticVerifier,
    CitationIntegrityChecker,
    ClaimVerificationResult,
    ClaimVerifierOrchestrator,
    InactiveSemanticVerifier,
    LLMClaimVerifier,
    PropositionAnalyzer,
    RuleBasedSemanticVerifier,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_index() -> DocumentIndex:
    """Multi-page synthetic index with controlled factual, exception, and contradictory passages."""
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


# ── Layer A: Citation Integrity Tests ─────────────────────────────────────────

def test_citation_integrity_valid_citation(synthetic_index: DocumentIndex):
    """Test 1: Existing evidence ID and exact quotation pass Layer A verification."""
    checker = CitationIntegrityChecker(synthetic_index)
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is True
    assert err_msg is None
    assert sanitized is not None
    assert sanitized.quote == ev.quote
    assert sanitized.pdf_page_index == 1


def test_citation_integrity_fabricated_chunk_id(synthetic_index: DocumentIndex):
    """Test 2: Citation referencing non-existent chunk ID fails Layer A."""
    checker = CitationIntegrityChecker(synthetic_index)
    ev = EvidenceItem(
        evidence_id="E99",
        source_file="regulations_handbook.pdf",
        chunk_id="nonexistent_chunk_id",
        pdf_page_index=1,
        quote="Some quote",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is False
    assert "non-existent chunk ID" in err_msg


def test_citation_integrity_mismatched_source_document(synthetic_index: DocumentIndex):
    """Test 3: Citation referencing wrong source document fails Layer A."""
    checker = CitationIntegrityChecker(synthetic_index)
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="unrelated_document.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is False
    assert "source file" in err_msg.lower()


def test_citation_integrity_incorrect_physical_page(synthetic_index: DocumentIndex):
    """Test 4: Citation claiming wrong physical PDF page fails Layer A."""
    checker = CitationIntegrityChecker(synthetic_index)
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=99,  # True page is 1
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is False
    assert "physical page" in err_msg.lower()


def test_citation_integrity_conflicting_printed_page(synthetic_index: DocumentIndex):
    """Test 5: Citation claiming conflicting printed page label fails Layer A."""
    checker = CitationIntegrityChecker(synthetic_index)
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        printed_page_number="page 999",  # Real metadata has "i"
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is False
    assert "printed page" in err_msg.lower()


def test_citation_integrity_fabricated_quotation(synthetic_index: DocumentIndex):
    """Test 6: Fabricated quote not in source chunk text fails Layer A."""
    checker = CitationIntegrityChecker(synthetic_index)
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="This sentence was completely fabricated by the model.",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is False
    assert "quotation rejected" in err_msg.lower()


def test_citation_integrity_quotation_from_different_chunk(synthetic_index: DocumentIndex):
    """Test 7: Quote that exists in chunk 3 attributed to chunk 1 fails Layer A."""
    checker = CitationIntegrityChecker(synthetic_index)
    # Quote is from chunk 3 on page 2, but cited as chunk 1 on page 1
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="Non-commercial specimens intended for non-profit research institutions are exempt from the standard permit fee.",
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is False
    assert "quotation rejected" in err_msg.lower()


def test_citation_integrity_altered_offsets_computed(synthetic_index: DocumentIndex):
    """Test 8: Valid quote has exact character offsets computed on physical page."""
    checker = CitationIntegrityChecker(synthetic_index)
    quote = "Species in the order PRIMATES are catalogued in Schedule B."
    ev = EvidenceItem(
        evidence_id="E2",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c01",
        pdf_page_index=1,
        quote=quote,
    )
    is_valid, err_msg, sanitized = checker.verify_evidence(ev)
    assert is_valid is True
    assert sanitized.page_char_start is not None
    assert sanitized.page_char_end is not None
    assert sanitized.page_char_end > sanitized.page_char_start


# ── Layer B & C: Semantic Claim Verification Tests ────────────────────────────

def test_semantic_verifier_direct_support(synthetic_index: DocumentIndex):
    """Test 9: Claim directly supported by the cited passage evaluates as verified_supported."""
    verifier = RuleBasedSemanticVerifier()
    claim = Claim(
        claim_id="C1",
        claim="Commercial exports of biological specimens require a Tier-1 permit.",
        evidence_ids=["E1"],
    )
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    res = verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.VERIFIED_SUPPORTED
    assert "directly supported" in res.explanation.lower()


def test_semantic_verifier_true_quote_unrelated_claim(synthetic_index: DocumentIndex):
    """
    Test 10 (Critical): A genuine quotation used to support a completely unrelated claim
    must evaluate as unsupported, proving exact quotation match alone does NOT establish support.
    """
    verifier = RuleBasedSemanticVerifier()
    # Claim is about corporate revenue
    claim = Claim(
        claim_id="C1",
        claim="The company generated 500 million dollars in annual software subscription revenue.",
        evidence_ids=["E1"],
    )
    # Evidence quote is genuine from Section 1 about biological permits
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    res = verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.UNSUPPORTED
    assert "does not address the claim topic" in res.explanation.lower()


def test_semantic_verifier_overgeneralization_detected(synthetic_index: DocumentIndex):
    """Test 11: Claim that overgeneralizes a qualified statement evaluates as unsupported."""
    verifier = RuleBasedSemanticVerifier()
    claim = Claim(
        claim_id="C1",
        claim="All research institutions are unconditionally exempt from permit fees without restriction.",
        evidence_ids=["E1"],
    )
    # Quote contains qualification "However... subject to health verification"
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p002_c00",
        pdf_page_index=2,
        quote=(
            "Non-commercial specimens intended for non-profit research institutions are exempt "
            "from the standard permit fee. However, research export permits remain strictly subject to health verification."
        ),
    )
    res = verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.UNSUPPORTED
    assert "overgeneralizes" in res.explanation.lower()


def test_semantic_verifier_contradiction_detected(synthetic_index: DocumentIndex):
    """Test 12: Claim contradicted by cited evidence evaluates as contradicted."""
    verifier = RuleBasedSemanticVerifier()
    claim = Claim(
        claim_id="C1",
        claim="International transfer of biological samples is permitted without presidential decree.",
        evidence_ids=["E1"],
    )
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p003_c00",
        pdf_page_index=3,
        quote="Notwithstanding any other provision, international transfer of biological samples is prohibited without presidential decree.",
    )
    res = verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.CONTRADICTED
    assert "contradicts" in res.explanation.lower()


def test_semantic_verifier_compound_statement_partially_unsupported(synthetic_index: DocumentIndex):
    """Test 13: Compound sentence where one proposition is supported and one is unsupported."""
    verifier = RuleBasedSemanticVerifier()
    # Proposition 1 (Tier-1 permit) is in evidence; Proposition 2 (export to Japan is authorized) is NOT
    claim = Claim(
        claim_id="C1",
        claim="Commercial exports of biological specimens require a Tier-1 permit; and immediate unrestricted export to Japan is authorized.",
        evidence_ids=["E1"],
    )
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    res = verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.UNSUPPORTED
    assert len(res.unsupported_propositions) > 0


def test_semantic_verifier_multi_passage_synthesis(synthetic_index: DocumentIndex):
    """Test 14: Conclusion requiring multiple passages is verified supported when all evidence is present."""
    verifier = RuleBasedSemanticVerifier()
    claim = Claim(
        claim_id="C1",
        claim="Commercial exports of biological specimens of primates require both a Tier-1 permit and temperature-controlled transit.",
        evidence_ids=["E1", "E2"],
        is_inference=True,
    )
    ev1 = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    ev2 = EvidenceItem(
        evidence_id="E2",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c01",
        pdf_page_index=1,
        quote="All Schedule B specimens require specialized temperature-controlled transit.",
    )
    res = verifier.verify_claim(claim, [ev1, ev2], [])
    assert res.status == ClaimVerificationStatus.VERIFIED_SUPPORTED


def test_llm_claim_verifier_malformed_json_fallback():
    """Test 15: LLM verifier returning malformed output safely evaluates as inconclusive."""
    def malformed_llm(sys_prompt: str, user_prompt: str) -> str:
        return "I am an LLM and cannot evaluate this: not json!"

    llm_verifier = LLMClaimVerifier(llm_callable=malformed_llm)
    claim = Claim(claim_id="C1", claim="Some claim", evidence_ids=["E1"])
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="test.pdf",
        chunk_id="c0",
        pdf_page_index=1,
        quote="Some quote",
    )
    res = llm_verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.INCONCLUSIVE
    assert "malformed output" in res.explanation.lower()


def test_inactive_semantic_verifier():
    """Test 16: Inactive semantic verifier explicitly returns not_checked."""
    verifier = InactiveSemanticVerifier()
    claim = Claim(claim_id="C1", claim="Some claim", evidence_ids=["E1"])
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="test.pdf",
        chunk_id="c0",
        pdf_page_index=1,
        quote="Some quote",
    )
    res = verifier.verify_claim(claim, [ev], [])
    assert res.status == ClaimVerificationStatus.NOT_CHECKED
    assert "not executed" in res.explanation.lower()


# ── End-to-End Orchestrator & QA Engine Tests ─────────────────────────────────

def test_orchestrator_contradicted_claim_downgrades_to_conflicting_evidence(synthetic_index: DocumentIndex):
    """Test 17: Contradicted claim causes response status to become conflicting_evidence."""
    claim = Claim(
        claim_id="C1",
        claim="Transfer of biological samples is permitted without presidential decree.",
        evidence_ids=["E1"],
    )
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p003_c00",
        pdf_page_index=3,
        quote="Notwithstanding any other provision, international transfer of biological samples is prohibited without presidential decree.",
    )
    draft_response = QAResponse(
        question="Is transfer permitted?",
        answer="Transfer is permitted.",
        status=QAStatus.ANSWERED.value,
        claims=[claim],
        evidence=[ev],
    )
    orchestrator = ClaimVerifierOrchestrator(synthetic_index)
    verified_response = orchestrator.verify_response(draft_response)

    assert verified_response.status == QAStatus.CONFLICTING_EVIDENCE.value
    assert verified_response.claims[0].verification_status == ClaimVerificationStatus.CONTRADICTED.value
    assert any("contradicted" in lim.lower() for lim in verified_response.limitations)


def test_orchestrator_unsupported_claim_downgrades_to_insufficient_evidence(synthetic_index: DocumentIndex):
    """Test 18: Answer where all claims are unsupported downgrades to insufficient_evidence."""
    # True quote, completely unrelated claim
    claim = Claim(
        claim_id="C1",
        claim="Company revenue expanded 200 percent across Europe.",
        evidence_ids=["E1"],
    )
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    draft_response = QAResponse(
        question="What was company revenue?",
        answer="Company revenue expanded.",
        status=QAStatus.ANSWERED.value,
        claims=[claim],
        evidence=[ev],
    )
    orchestrator = ClaimVerifierOrchestrator(synthetic_index)
    verified_response = orchestrator.verify_response(draft_response)

    assert verified_response.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert verified_response.claims[0].verification_status == ClaimVerificationStatus.UNSUPPORTED.value
    assert any("unsupported" in lim.lower() or "none" in lim.lower() for lim in verified_response.limitations)


def test_orchestrator_partial_support_downgrades_to_partially_answered(synthetic_index: DocumentIndex):
    """Test 19: Answer with one supported and one unsupported claim becomes partially_answered."""
    c1 = Claim(
        claim_id="C1",
        claim="All commercial exports of biological specimens require a Tier-1 permit.",
        evidence_ids=["E1"],
    )
    c2 = Claim(
        claim_id="C2",
        claim="Global pharmaceutical shipments generate 80 percent of total profits.",
        evidence_ids=["E1"],  # Unrelated to E1
    )
    ev = EvidenceItem(
        evidence_id="E1",
        source_file="regulations_handbook.pdf",
        chunk_id="test_doc_001_p001_c00",
        pdf_page_index=1,
        quote="All commercial exports of biological specimens require a valid Tier-1 permit issued by the Wildlife Authority.",
    )
    draft_response = QAResponse(
        question="What are the permit rules and profit numbers?",
        answer="Tier-1 permit is required and shipments generate 80% profit.",
        status=QAStatus.ANSWERED.value,
        claims=[c1, c2],
        evidence=[ev],
    )
    orchestrator = ClaimVerifierOrchestrator(synthetic_index)
    verified_response = orchestrator.verify_response(draft_response)

    assert verified_response.status == QAStatus.PARTIALLY_ANSWERED.value
    assert verified_response.claims[0].verification_status == ClaimVerificationStatus.VERIFIED_SUPPORTED.value
    assert verified_response.claims[1].verification_status == ClaimVerificationStatus.UNSUPPORTED.value


def test_cli_verification_output_subprocess(tmp_path: Path, synthetic_index: DocumentIndex):
    """Test 20: CLI output displays claim-level verification statuses and audit details."""
    index_file = tmp_path / "test_cli_verify_index.json"
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
    assert proc.returncode == 0, f"CLI error: {proc.stderr}"
    stdout = proc.stdout
    assert "EVALUATED CLAIMS & VERIFICATION STATUS:" in stdout
    assert "Status : [" in stdout
    assert "via rule_based" in stdout


def test_cites_proof_of_concept_with_claim_verification():
    """
    Test 21: Full proof-of-concept verification on CITES document with claim-level validation.
    """
    index_path = Path("outputs/cites_index.json")
    pdf_path = Path("CITES_E-Appendices.pdf")

    if not index_path.exists() and pdf_path.exists():
        from graphrag.ingestion.indexer import index_pdf
        index = index_pdf(pdf_path, output_path=index_path)
    elif index_path.exists():
        index = DocumentIndex.load(index_path)
    else:
        pytest.skip("CITES document index not available.")

    engine = DocumentQAEngine(mock=True)

    # Q1: Primate export for pharmaceutical research
    q1 = "Can Rhesus Monkeys be exported to the USA for pharmaceutical research?"
    resp1 = engine.answer_question(index, q1)
    assert resp1.status == QAStatus.PARTIALLY_ANSWERED.value
    assert len(resp1.claims) > 0
    # The taxonomy claim must be verified supported
    assert resp1.claims[0].verification_status == ClaimVerificationStatus.VERIFIED_SUPPORTED.value
    assert any("pharmaceutical" in lim.lower() or "research" in lim.lower() for lim in resp1.limitations)

    # Q2: Specific Indian government ministries and approvals
    q2 = "Are specific Indian central-government approvals required for this export, and which approvals or ministries are identified in the supplied document?"
    resp2 = engine.answer_question(index, q2)
    assert resp2.status == QAStatus.INSUFFICIENT_EVIDENCE.value
    assert len(resp2.claims) == 0
    assert any("indian" in lim.lower() or "ministr" in lim.lower() for lim in resp2.limitations)
