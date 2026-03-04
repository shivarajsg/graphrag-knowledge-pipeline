"""
graphrag.qa_verifier
--------------------
Phase 3: Claim-Level Evidence Verification and Hallucination Safeguards.
Implements layered verification:
  - Layer A: Deterministic Citation Integrity
  - Layer B: Proposition Completeness & Exception Analysis
  - Layer C: Semantic Claim-to-Evidence Verification
Integrates evidence sufficiency and abstention behavior to safeguard answer factuality.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .ingestion.indexer import DocumentSearchResult
from .ingestion.schemas import DocumentChunk, DocumentIndex
from .qa_schemas import Claim, ClaimVerificationStatus, EvidenceItem, QAResponse, QAStatus
from .qa_validator import _normalize_text, find_quote_in_chunk
from .utils import safe_parse_json

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerificationResult:
    """Detailed result for an individual claim verification evaluation."""
    claim_id: str
    status: ClaimVerificationStatus
    explanation: str
    verification_method: str
    evidence_ids: List[str]
    contradiction_details: Optional[str] = None
    unsupported_propositions: List[str] = field(default_factory=list)


# ── Layer A: Deterministic Citation Integrity ─────────────────────────────────

class CitationIntegrityChecker:
    """
    Deterministically validates that evidence citations reference valid,
    authentic, unmanipulated passages from the active document index.
    """

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self._chunks_by_id: Dict[str, DocumentChunk] = {c.chunk_id: c for c in index.chunks}

    def verify_evidence(
        self,
        evidence: EvidenceItem,
    ) -> Tuple[bool, Optional[str], Optional[EvidenceItem]]:
        """
        Verify an individual evidence item against the index.
        Returns: (is_valid, error_message, sanitized_evidence_item)
        """
        eid = evidence.evidence_id
        chunk = self._chunks_by_id.get(evidence.chunk_id)

        # 1. Chunk existence
        if chunk is None:
            return (
                False,
                f"Evidence '{eid}' references non-existent chunk ID '{evidence.chunk_id}'.",
                None,
            )

        # 2. Source document identity
        if evidence.source_file and evidence.source_file != chunk.source_file:
            return (
                False,
                f"Evidence '{eid}' source file '{evidence.source_file}' does not match "
                f"chunk source file '{chunk.source_file}'.",
                None,
            )

        # 3. Physical PDF page
        if evidence.pdf_page_index != chunk.pdf_page_index:
            return (
                False,
                f"Evidence '{eid}' physical page {evidence.pdf_page_index} does not match "
                f"chunk '{chunk.chunk_id}' physical page {chunk.pdf_page_index}.",
                None,
            )

        # 4. Printed page
        if evidence.printed_page_number and chunk.printed_page_number:
            if evidence.printed_page_number != chunk.printed_page_number:
                return (
                    False,
                    f"Evidence '{eid}' printed page '{evidence.printed_page_number}' conflicts with "
                    f"chunk metadata '{chunk.printed_page_number}'.",
                    None,
                )

        # 5. Quotation authenticity
        found, exact_quote, char_start, char_end = find_quote_in_chunk(evidence.quote, chunk.text)
        if not found:
            preview = evidence.quote[:60] + ("…" if len(evidence.quote) > 60 else "")
            return (
                False,
                f"Evidence '{eid}' quotation rejected: quote \"{preview}\" was not found in "
                f"source chunk '{chunk.chunk_id}' on physical page {chunk.pdf_page_index}.",
                None,
            )

        # 6. Verify / compute absolute page character offsets
        if char_start is not None and char_end is not None:
            abs_start = chunk.page_char_start + char_start
            abs_end = chunk.page_char_start + char_end
        else:
            abs_start = chunk.page_char_start
            abs_end = chunk.page_char_end

        # Create verified evidence item
        verified_item = EvidenceItem(
            evidence_id=evidence.evidence_id,
            source_file=chunk.source_file,
            chunk_id=chunk.chunk_id,
            pdf_page_index=chunk.pdf_page_index,
            quote=(
                evidence.quote
                if (evidence.quote and _normalize_text(evidence.quote) == _normalize_text(exact_quote))
                else (exact_quote or evidence.quote)
            ),
            printed_page_number=chunk.printed_page_number or evidence.printed_page_number,
            section_heading=chunk.section_heading or evidence.section_heading,
            page_char_start=abs_start,
            page_char_end=abs_end,
            document_id=chunk.document_id,
            extraction_method=chunk.extraction_method,
        )

        return True, None, verified_item


# ── Layer B: Proposition & Completeness Analyzer ──────────────────────────────

class PropositionAnalyzer:
    """
    Splits compound claims into discrete assertions and detects overgeneralizations
    or omitted material exceptions.
    """

    @staticmethod
    def split_into_propositions(claim_text: str) -> List[str]:
        """
        Split a compound claim sentence into distinct assertions if it joins
        multiple independent clauses with conjunctions or semicolons.
        """
        # Split on semicolons first
        clauses = [c.strip() for c in claim_text.split(";") if c.strip()]
        propositions = []

        for clause in clauses:
            # Check for strong conjunction splitting: "and also", "and therefore", "and export", etc.
            # Only split if both sides contain verb forms
            parts = re.split(r",\s*(?:and|while|whereas|but)\s+|\s+and\s+(?=[a-z\s]+(?:is|are|require|can|must|will))", clause, flags=re.IGNORECASE)
            for part in parts:
                p_clean = part.strip()
                if p_clean:
                    propositions.append(p_clean)

        return propositions if propositions else [claim_text.strip()]

    @staticmethod
    def check_overgeneralization_and_exceptions(
        claim_text: str,
        evidence_texts: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects if a claim overgeneralizes a qualified statement or omits a crucial exception.
        Returns: (has_flaw, flaw_description)
        """
        claim_lower = claim_text.lower()
        combined_evidence = " ".join(evidence_texts).lower()

        # Exception pattern: Evidence contains "except", "unless", "excluding", "subject to"
        exception_indicators = ["except for", "except", "unless", "excluding", "subject to health", "subject to quota"]
        for exc in exception_indicators:
            if exc in combined_evidence:
                # Check if claim acknowledges the exception
                claim_has_exception = any(w in claim_lower for w in ["except", "unless", "excluding", "subject to", "with exception"])
                # Check if claim uses absolute terms
                claim_uses_absolute = any(w in claim_lower for w in ["all ", "any ", "always", "without restriction", "unconditionally", "permitted everywhere"])
                if claim_uses_absolute and not claim_has_exception:
                    return (
                        True,
                        f"Claim overgeneralizes with absolute rule ('{claim_text[:50]}…') but source passage contains exception: '{exc}'.",
                    )

        return False, None


# ── Layer C: Semantic Verifier Backends ───────────────────────────────────────

class BaseSemanticVerifier:
    """Base interface for semantic claim-to-evidence verification."""

    def verify_claim(
        self,
        claim: Claim,
        cited_evidence: List[EvidenceItem],
        all_passages: List[DocumentSearchResult],
    ) -> ClaimVerificationResult:
        raise NotImplementedError


class RuleBasedSemanticVerifier(BaseSemanticVerifier):
    """
    Deterministic rule-based semantic verifier.
    Evaluates:
      1. Topical alignment (detects true quotes attached to completely unrelated claims).
      2. Negation / polarity contradictions.
      3. Overgeneralization / exception omission.
    """

    def verify_claim(
        self,
        claim: Claim,
        cited_evidence: List[EvidenceItem],
        all_passages: List[DocumentSearchResult],
    ) -> ClaimVerificationResult:
        cid = claim.claim_id or "C1"
        c_text = claim.claim
        c_lower = c_text.lower()

        if not cited_evidence:
            return ClaimVerificationResult(
                claim_id=cid,
                status=ClaimVerificationStatus.UNSUPPORTED,
                explanation="No cited evidence provided to support this claim.",
                verification_method="rule_based",
                evidence_ids=[],
            )

        evidence_quotes = [e.quote for e in cited_evidence]
        combined_quotes = " ".join(evidence_quotes).lower()

        # ── Check 1: Unrelated Claim Detection (True Quote, False/Unrelated Claim) ──
        # Extract meaningful nouns/terms from claim
        stopwords = {
            "what", "which", "when", "where", "does", "state", "about", "all", "the",
            "and", "for", "are", "with", "from", "that", "this", "also", "into", "their",
            "order", "have", "has", "been", "according", "document", "specifies",
        }
        claim_terms = set(re.findall(r"\b[a-z]{3,}\b", c_lower)) - stopwords
        if claim_terms:
            overlap = [t for t in claim_terms if t in combined_quotes]
            overlap_ratio = len(overlap) / len(claim_terms)
            # If fewer than 20% of claim's key terms appear in cited evidence quote
            if overlap_ratio < 0.20 and len(claim_terms) >= 3:
                return ClaimVerificationResult(
                    claim_id=cid,
                    status=ClaimVerificationStatus.UNSUPPORTED,
                    explanation=(
                        f"Cited passage quote does not address the claim topic. "
                        f"Key terms {list(claim_terms - set(overlap))[:3]} not found in cited quote."
                    ),
                    verification_method="rule_based",
                    evidence_ids=[e.evidence_id for e in cited_evidence],
                )

        # ── Check 2: Contradiction Detection ──
        # If claim says "permitted" or "allowed", but quote says "prohibited" or "forbidden"
        pos_words = {"permitted", "allowed", "authorized", "exempt", "legal"}
        neg_words = {"prohibited", "forbidden", "illegal", "unauthorized", "strictly prohibited"}

        claim_has_pos = any(w in c_lower for w in pos_words)
        claim_has_neg = any(w in c_lower for w in neg_words)
        quote_has_pos = any(w in combined_quotes for w in pos_words)
        quote_has_neg = any(w in combined_quotes for w in neg_words)

        if (claim_has_pos and quote_has_neg and not quote_has_pos) or (claim_has_neg and quote_has_pos and not quote_has_neg):
            return ClaimVerificationResult(
                claim_id=cid,
                status=ClaimVerificationStatus.CONTRADICTED,
                explanation="The cited source evidence explicitly contradicts the claim polarity.",
                verification_method="rule_based",
                evidence_ids=[e.evidence_id for e in cited_evidence],
                contradiction_details=f"Claim polarity conflicts with source quote text: '{evidence_quotes[0][:80]}…'",
            )

        # ── Check 3: Overgeneralization & Omitted Exceptions ──
        has_flaw, flaw_msg = PropositionAnalyzer.check_overgeneralization_and_exceptions(
            claim_text=c_text,
            evidence_texts=evidence_quotes,
        )
        if has_flaw:
            return ClaimVerificationResult(
                claim_id=cid,
                status=ClaimVerificationStatus.UNSUPPORTED,
                explanation=flaw_msg or "Claim overgeneralizes without citing required condition.",
                verification_method="rule_based",
                evidence_ids=[e.evidence_id for e in cited_evidence],
            )

        # ── Check 4: Compound Proposition Completeness ──
        propositions = PropositionAnalyzer.split_into_propositions(c_text)
        if len(propositions) > 1:
            unsupported_props = []
            quote_words = set(re.findall(r"\b[a-z]{3,}\b", combined_quotes))
            for prop in propositions:
                prop_terms = set(re.findall(r"\b[a-z]{3,}\b", prop.lower())) - stopwords
                prop_overlap = [t for t in prop_terms if t in quote_words]
                overlap_ratio = len(prop_overlap) / len(prop_terms) if prop_terms else 1.0
                if prop_terms and overlap_ratio < 0.50:
                    unsupported_props.append(prop)

            if unsupported_props:
                return ClaimVerificationResult(
                    claim_id=cid,
                    status=ClaimVerificationStatus.UNSUPPORTED,
                    explanation=(
                        f"Compound claim is partially unsupported: proposition(s) "
                        f"{unsupported_props} are not established by the cited evidence."
                    ),
                    verification_method="rule_based",
                    evidence_ids=[e.evidence_id for e in cited_evidence],
                    unsupported_propositions=unsupported_props,
                )


        # ── Supported ──
        return ClaimVerificationResult(
            claim_id=cid,
            status=ClaimVerificationStatus.VERIFIED_SUPPORTED,
            explanation=f"Claim is directly supported by cited evidence passage(s): {', '.join(e.evidence_id for e in cited_evidence)}.",
            verification_method="rule_based",
            evidence_ids=[e.evidence_id for e in cited_evidence],
        )


class LLMClaimVerifier(BaseSemanticVerifier):
    """
    LLM-powered semantic claim verifier using a narrowly scoped verification prompt.
    """

    def __init__(self, llm_callable: Callable[[str, str], str], model_name: str = "claude-sonnet-5-5") -> None:
        self.llm_callable = llm_callable
        self.model_name = model_name

    def verify_claim(
        self,
        claim: Claim,
        cited_evidence: List[EvidenceItem],
        all_passages: List[DocumentSearchResult],
    ) -> ClaimVerificationResult:
        cid = claim.claim_id or "C1"
        if not cited_evidence:
            return ClaimVerificationResult(
                claim_id=cid,
                status=ClaimVerificationStatus.UNSUPPORTED,
                explanation="No evidence passages cited.",
                verification_method="llm_verifier",
                evidence_ids=[],
            )

        evidence_str = "\n".join(
            f"[{e.evidence_id}] (Page {e.pdf_page_index}): \"{e.quote}\"" for e in cited_evidence
        )

        system_prompt = (
            "You are a rigorous factual claim verification judge. Your task is to evaluate whether "
            "the provided CLAIM is strictly supported, unsupported, or contradicted by the CITED EVIDENCE.\n\n"
            "Status Definitions:\n"
            "- 'verified_supported': The cited evidence directly establishes or logically entails the claim.\n"
            "- 'unsupported': The cited evidence fails to prove the claim, or the claim asserts facts not in the evidence.\n"
            "- 'contradicted': The cited evidence directly refutes the claim.\n"
            "- 'inconclusive': The cited evidence is too ambiguous to determine.\n\n"
            "Output JSON only: {\"status\": \"verified_supported|unsupported|contradicted|inconclusive\", \"explanation\": \"...\"}"
        )
        user_prompt = f"CLAIM:\n{claim.claim}\n\nCITED EVIDENCE:\n{evidence_str}\n\nEvaluate claim support strictly against evidence."

        try:
            raw = self.llm_callable(system_prompt, user_prompt)
            data = safe_parse_json(raw)
            if not data or "status" not in data:
                return ClaimVerificationResult(
                    claim_id=cid,
                    status=ClaimVerificationStatus.INCONCLUSIVE,
                    explanation="Verification model returned malformed output.",
                    verification_method="llm_verifier",
                    evidence_ids=[e.evidence_id for e in cited_evidence],
                )

            raw_status = str(data["status"]).lower()
            status_map = {
                "verified_supported": ClaimVerificationStatus.VERIFIED_SUPPORTED,
                "supported": ClaimVerificationStatus.VERIFIED_SUPPORTED,
                "unsupported": ClaimVerificationStatus.UNSUPPORTED,
                "contradicted": ClaimVerificationStatus.CONTRADICTED,
                "inconclusive": ClaimVerificationStatus.INCONCLUSIVE,
            }
            final_status = status_map.get(raw_status, ClaimVerificationStatus.INCONCLUSIVE)
            return ClaimVerificationResult(
                claim_id=cid,
                status=final_status,
                explanation=str(data.get("explanation", "")),
                verification_method=f"llm_verifier ({self.model_name})",
                evidence_ids=[e.evidence_id for e in cited_evidence],
            )
        except Exception as exc:
            logger.error(f"[verifier] LLM verification call failed: {exc}")
            return ClaimVerificationResult(
                claim_id=cid,
                status=ClaimVerificationStatus.INCONCLUSIVE,
                explanation=f"Semantic verification failed with exception: {exc}",
                verification_method="llm_verifier",
                evidence_ids=[e.evidence_id for e in cited_evidence],
            )


class InactiveSemanticVerifier(BaseSemanticVerifier):
    """Placeholder when semantic verification is intentionally not executed."""

    def verify_claim(
        self,
        claim: Claim,
        cited_evidence: List[EvidenceItem],
        all_passages: List[DocumentSearchResult],
    ) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id or "C1",
            status=ClaimVerificationStatus.NOT_CHECKED,
            explanation="Semantic verification backend was not executed or not configured.",
            verification_method="none",
            evidence_ids=[e.evidence_id for e in cited_evidence],
        )


# ── Verification Orchestrator ─────────────────────────────────────────────────

class ClaimVerifierOrchestrator:
    """
    Coordinates Layer A (Citation Integrity), Layer B (Proposition Completeness),
    and Layer C (Semantic Claim-to-Evidence Verification) across all claims in a QAResponse.
    """

    def __init__(
        self,
        index: DocumentIndex,
        semantic_verifier: Optional[BaseSemanticVerifier] = None,
    ) -> None:
        self.index = index
        self.citation_checker = CitationIntegrityChecker(index)
        self.semantic_verifier = semantic_verifier or RuleBasedSemanticVerifier()

    def verify_response(
        self,
        response: QAResponse,
        retrieved_passages: Optional[List[DocumentSearchResult]] = None,
    ) -> QAResponse:
        """
        Runs comprehensive claim-level verification and updates response status,
        claims, limitations, and auditable evidence trail.
        """
        all_passages = retrieved_passages or []
        warnings: List[str] = list(response.validation_warnings)
        limitations: List[str] = list(response.limitations)
        evidence_trail: List[str] = list(response.evidence_trail)

        # ── Step 1: Layer A - Check Citation Integrity for all evidence items ──
        valid_evidence_items: List[EvidenceItem] = []
        valid_evidence_by_id: Dict[str, EvidenceItem] = {}

        for ev in response.evidence:
            is_valid, err_msg, sanitized_ev = self.citation_checker.verify_evidence(ev)
            if not is_valid:
                logger.warning(f"[verifier:LayerA] {err_msg}")
                warnings.append(f"Citation Integrity: {err_msg}")
            else:
                assert sanitized_ev is not None
                valid_evidence_items.append(sanitized_ev)
                valid_evidence_by_id[sanitized_ev.evidence_id] = sanitized_ev

        # ── Step 2: Layer B & C - Verify Each Claim Independently ───────────────
        verified_claims: List[Claim] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0
        not_checked_count = 0

        for idx, claim in enumerate(response.claims, 1):
            claim_id = claim.claim_id or f"C{idx}"
            claim.claim_id = claim_id

            # Check if cited evidence IDs exist in valid evidence items
            cited_items: List[EvidenceItem] = []
            for eid in claim.evidence_ids:
                if eid in valid_evidence_by_id:
                    cited_items.append(valid_evidence_by_id[eid])
                else:
                    msg = f"Claim '{claim_id}' cited evidence ID '{eid}' which failed citation integrity or does not exist."
                    warnings.append(f"Citation Integrity: {msg}")

            if not cited_items:
                # Failed Layer A
                claim.verification_status = ClaimVerificationStatus.UNSUPPORTED.value
                claim.explanation = "All cited evidence IDs failed citation integrity verification."
                claim.verification_method = "citation_integrity"
                unsupported_count += 1
                verified_claims.append(claim)
                continue

            # Run Layer C (Semantic Verification)
            v_res = self.semantic_verifier.verify_claim(
                claim=claim,
                cited_evidence=cited_items,
                all_passages=all_passages,
            )

            claim.verification_status = v_res.status.value
            claim.explanation = v_res.explanation
            claim.verification_method = v_res.verification_method
            claim.contradiction_details = v_res.contradiction_details
            claim.unsupported_propositions = v_res.unsupported_propositions

            if v_res.status == ClaimVerificationStatus.VERIFIED_SUPPORTED:
                supported_count += 1
            elif v_res.status == ClaimVerificationStatus.UNSUPPORTED:
                unsupported_count += 1
            elif v_res.status == ClaimVerificationStatus.CONTRADICTED:
                contradicted_count += 1
            elif v_res.status == ClaimVerificationStatus.NOT_CHECKED:
                not_checked_count += 1

            verified_claims.append(claim)

        # ── Step 3: Determine Final Response Status from Verification Outcomes ─
        final_status = response.status

        if response.status == QAStatus.CONFLICTING_EVIDENCE.value or contradicted_count > 0:
            final_status = QAStatus.CONFLICTING_EVIDENCE.value
            if contradicted_count > 0:
                limitations.append(
                    f"{contradicted_count} material claim(s) were contradicted by the source evidence."
                )
        elif len(verified_claims) > 0 and supported_count == 0:
            final_status = QAStatus.INSUFFICIENT_EVIDENCE.value

            limitations.append(
                "None of the factual claims could be verified as supported by the document evidence."
            )
        elif len(verified_claims) > 0 and unsupported_count > 0 and supported_count > 0:
            final_status = QAStatus.PARTIALLY_ANSWERED.value
            limitations.append(
                f"{unsupported_count} of {len(verified_claims)} claims were unsupported by the cited source material."
            )
        elif not_checked_count > 0:
            limitations.append(
                "Semantic claim-to-evidence verification was not executed for one or more claims."
            )

        # ── Step 4: Build Auditable Evidence Trail ────────────────────────────
        audit_trail: List[str] = []
        for c in verified_claims:
            v_tag = c.verification_status.upper()
            e_str = ", ".join(c.evidence_ids) if c.evidence_ids else "none"
            audit_trail.append(
                f"Claim {c.claim_id} [{v_tag}]: \"{c.claim}\" (Evidence: {e_str}) — {c.explanation}"
            )

        return QAResponse(
            question=response.question,
            answer=response.answer,
            status=final_status,
            claims=verified_claims,
            evidence=valid_evidence_items,
            evidence_trail=audit_trail if audit_trail else evidence_trail,
            limitations=limitations,
            validation_warnings=warnings,
            metadata=dict(response.metadata),
        )
