"""
graphrag.qa_schemas
-------------------
Data models and response schemas for evidence-grounded question answering (Phase 2).
Provides structured definitions for claims, evidence provenance, and auditable answers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QAStatus(str, Enum):
    """Controlled status vocabulary for question answering."""
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS = "ambiguous"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    PROCESSING_ERROR = "processing_error"


@dataclass
class EvidenceItem:
    """
    Structured source evidence item linking a claim to verbatim document text.
    """
    evidence_id: str                          # Stable identifier within response, e.g. "E1"
    source_file: str                          # Original document filename
    chunk_id: str                             # Deterministic chunk ID in the index
    pdf_page_index: int                       # 1-based physical PDF page number
    quote: str                                # Verbatim excerpt from the indexed source chunk
    printed_page_number: Optional[str] = None # Printed page label if available
    section_heading: Optional[str] = None     # Section heading context
    page_char_start: Optional[int] = None     # Starting character offset on page
    page_char_end: Optional[int] = None       # Ending character offset on page
    document_id: Optional[str] = None         # Stable document identifier
    extraction_method: str = "native_text"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceItem:
        return cls(
            evidence_id=str(data.get("evidence_id", "")),
            source_file=str(data.get("source_file", "")),
            chunk_id=str(data.get("chunk_id", "")),
            pdf_page_index=int(data.get("pdf_page_index", 1)),
            quote=str(data.get("quote", "")),
            printed_page_number=data.get("printed_page_number"),
            section_heading=data.get("section_heading"),
            page_char_start=data.get("page_char_start"),
            page_char_end=data.get("page_char_end"),
            document_id=data.get("document_id"),
            extraction_method=data.get("extraction_method", "native_text"),
        )


class ClaimVerificationStatus(str, Enum):
    """
    Explicit verification status for an individual factual claim (Phase 3).
    """
    VERIFIED_SUPPORTED = "verified_supported"  # Evidence establishes the claim
    UNSUPPORTED = "unsupported"                # Evidence fails to establish the claim
    CONTRADICTED = "contradicted"              # Evidence directly contradicts the claim
    INCONCLUSIVE = "inconclusive"              # Evidence is ambiguous or insufficient to judge
    NOT_CHECKED = "not_checked"                # Verification was not executed or unavailable


@dataclass
class Claim:
    """
    A specific factual statement derived from document evidence.
    Supports claim-level verification tracking (Phase 3).
    """
    claim: str                                # The factual statement
    evidence_ids: List[str]                   # Evidence identifiers supporting this statement
    is_inference: bool = False                # True if inferred by synthesizing multiple passages
    claim_id: str = ""                        # Stable identifier within response, e.g. "C1", "C2"
    verification_status: str = ClaimVerificationStatus.NOT_CHECKED.value
    explanation: str = ""                     # Concise explanation of verification result
    verification_method: str = "none"         # e.g. deterministic_citation | semantic_nli | llm_verifier | rule_based
    contradiction_details: Optional[str] = None
    unsupported_propositions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "evidence_ids": list(self.evidence_ids),
            "is_inference": self.is_inference,
            "verification_status": self.verification_status,
            "explanation": self.explanation,
            "verification_method": self.verification_method,
            "contradiction_details": self.contradiction_details,
            "unsupported_propositions": list(self.unsupported_propositions),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Claim:
        return cls(
            claim_id=str(data.get("claim_id", "")),
            claim=str(data.get("claim", "")),
            evidence_ids=[str(e) for e in data.get("evidence_ids", [])],
            is_inference=bool(data.get("is_inference", False)),
            verification_status=str(data.get("verification_status", ClaimVerificationStatus.NOT_CHECKED.value)),
            explanation=str(data.get("explanation", "")),
            verification_method=str(data.get("verification_method", "none")),
            contradiction_details=data.get("contradiction_details"),
            unsupported_propositions=list(data.get("unsupported_propositions", [])),
        )


@dataclass
class QAResponse:
    """
    Complete, validated, evidence-grounded question-answering response.
    """
    question: str
    answer: str
    status: str                               # Value from QAStatus
    claims: List[Claim] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    evidence_trail: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "claims": [c.to_dict() for c in self.claims],
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_trail": list(self.evidence_trail),
            "limitations": list(self.limitations),
            "validation_warnings": list(self.validation_warnings),
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAResponse:
        claims = [Claim.from_dict(c) for c in data.get("claims", [])]
        evidence = [EvidenceItem.from_dict(e) for e in data.get("evidence", [])]
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            status=str(data.get("status", QAStatus.INSUFFICIENT_EVIDENCE.value)),
            claims=claims,
            evidence=evidence,
            evidence_trail=list(data.get("evidence_trail", [])),
            limitations=list(data.get("limitations", [])),
            validation_warnings=list(data.get("validation_warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def get_evidence_by_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        for e in self.evidence:
            if e.evidence_id == evidence_id:
                return e
        return None
