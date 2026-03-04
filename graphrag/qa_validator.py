"""
graphrag.qa_validator
---------------------
Deterministic validation and integrity verification for citations, quotations,
page provenance, and response schema compliance (Phase 2).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .ingestion.schemas import DocumentChunk, DocumentIndex
from .qa_schemas import Claim, EvidenceItem, QAResponse, QAStatus

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize whitespace for resilient substring comparison."""
    return re.sub(r"\s+", " ", text).strip()


def find_quote_in_chunk(quote: str, chunk_text: str) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """
    Search for a quoted passage within a chunk's text.
    First tries exact match. If not found, tries normalized whitespace.
    Returns (found, exact_slice, char_start, char_end).
    """
    q_clean = quote.strip()
    if not q_clean:
        return False, None, None, None

    # 1. Exact match
    idx = chunk_text.find(q_clean)
    if idx != -1:
        return True, q_clean, idx, idx + len(q_clean)

    # 2. Case-insensitive exact match
    idx_lower = chunk_text.lower().find(q_clean.lower())
    if idx_lower != -1:
        exact_slice = chunk_text[idx_lower : idx_lower + len(q_clean)]
        return True, exact_slice, idx_lower, idx_lower + len(q_clean)

    # 3. Whitespace-normalized match
    # Strip common leading/trailing quotes if present
    q_unquoted = q_clean.strip("\"'“'”")
    if q_unquoted:
        idx_unquoted = chunk_text.find(q_unquoted)
        if idx_unquoted != -1:
            return True, q_unquoted, idx_unquoted, idx_unquoted + len(q_unquoted)

    # Check normalized text match
    norm_chunk = _normalize_text(chunk_text)
    norm_quote = _normalize_text(q_clean)
    if norm_quote and norm_quote.lower() in norm_chunk.lower():
        # Substring exists semantically in text despite whitespace variations.
        # Try to locate the span boundaries in the original text using strict word sequence
        words = norm_quote.split()
        if words:
            # 1. First try strict word sequence with whitespace/newlines between words
            strict_pattern = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
            match = strict_pattern.search(chunk_text)
            if match:
                exact_slice = match.group(0)
                return True, exact_slice, match.start(), match.end()

            # 2. If len(words) >= 2, find candidate matches whose normalized text matches norm_quote
            if len(words) >= 2:
                first_word = re.escape(words[0])
                last_word = re.escape(words[-1])
                pattern = re.compile(rf"{first_word}[\s\S]*?{last_word}", re.IGNORECASE)
                for m in pattern.finditer(chunk_text):
                    candidate = m.group(0)
                    if _normalize_text(candidate).lower() == norm_quote.lower():
                        return True, candidate, m.start(), m.end()

    return False, None, None, None


class CitationValidator:
    """
    Validates citation provenance, quotation authenticity, and schema integrity
    against an active DocumentIndex.
    """

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self._chunks_by_id: Dict[str, DocumentChunk] = {c.chunk_id: c for c in index.chunks}

    def validate_and_sanitize(
        self,
        response: QAResponse,
    ) -> QAResponse:
        """
        Validate and sanitize a QAResponse against the indexed document:
        - Checks that every evidence item references a real chunk.
        - Checks that physical page metadata matches the indexed chunk.
        - Verifies that quoted text actually exists in the source chunk.
        - Rejects fabricated quotations and ungrounded claims.
        - Checks that claims reference real, validated evidence IDs.
        - Ensures appropriate response status after validation.
        """
        warnings: List[str] = list(response.validation_warnings)
        valid_evidence_items: List[EvidenceItem] = []
        valid_evidence_ids: Set[str] = set()

        # ── Step 1: Validate Evidence Items ───────────────────────────────────
        for ev in response.evidence:
            eid = ev.evidence_id
            chunk = self._chunks_by_id.get(ev.chunk_id)

            if chunk is None:
                msg = f"Evidence '{eid}' references non-existent chunk ID '{ev.chunk_id}'."
                logger.warning(f"[validator] {msg}")
                warnings.append(msg)
                continue

            # Verify and fix source file provenance
            if ev.source_file and ev.source_file != chunk.source_file:
                msg = (
                    f"Evidence '{eid}' source file '{ev.source_file}' does not match "
                    f"index source file '{chunk.source_file}'. Correcting."
                )
                logger.warning(f"[validator] {msg}")
                warnings.append(msg)
                ev.source_file = chunk.source_file

            # Verify physical page index
            if ev.pdf_page_index != chunk.pdf_page_index:
                msg = (
                    f"Evidence '{eid}' cited physical page {ev.pdf_page_index}, "
                    f"but chunk '{chunk.chunk_id}' is on physical page {chunk.pdf_page_index}. Correcting."
                )
                logger.warning(f"[validator] {msg}")
                warnings.append(msg)
                ev.pdf_page_index = chunk.pdf_page_index

            # Populate printed page if known in chunk
            if chunk.printed_page_number and not ev.printed_page_number:
                ev.printed_page_number = chunk.printed_page_number

            # Verify section heading if available in chunk
            if chunk.section_heading and (not ev.section_heading or ev.section_heading.upper() in chunk.text.upper()):
                ev.section_heading = chunk.section_heading

            # Verify quotation against chunk text
            found, exact_quote, char_start, char_end = find_quote_in_chunk(ev.quote, chunk.text)
            if not found:
                preview = ev.quote[:60] + ("…" if len(ev.quote) > 60 else "")
                msg = (
                    f"Evidence '{eid}' quotation rejected: quote \"{preview}\" "
                    f"was not found in source chunk '{chunk.chunk_id}' (page {chunk.pdf_page_index})."
                )
                logger.warning(f"[validator] {msg}")
                warnings.append(msg)
                # Discard fabricated or invalid quote
                continue

            # Quotation verified! Update to exact text slice and compute absolute page offsets
            if not ev.quote or _normalize_text(ev.quote) != _normalize_text(exact_quote):
                ev.quote = exact_quote or ev.quote
            if char_start is not None and char_end is not None:
                ev.page_char_start = chunk.page_char_start + char_start
                ev.page_char_end = chunk.page_char_start + char_end
            else:
                ev.page_char_start = chunk.page_char_start
                ev.page_char_end = chunk.page_char_end

            valid_evidence_items.append(ev)
            valid_evidence_ids.add(eid)

        # ── Step 2: Validate Claims ───────────────────────────────────────────
        valid_claims: List[Claim] = []
        discarded_claims_count = 0

        for claim in response.claims:
            orig_eids = list(claim.evidence_ids)
            filtered_eids = [eid for eid in orig_eids if eid in valid_evidence_ids]

            # Detect fabricated or dropped evidence IDs
            invalid_eids = set(orig_eids) - valid_evidence_ids
            if invalid_eids:
                msg = (
                    f"Claim \"{claim.claim[:60]}…\" cited invalid/rejected evidence IDs: "
                    f"{sorted(invalid_eids)}. Removing invalid IDs."
                )
                logger.warning(f"[validator] {msg}")
                warnings.append(msg)

            if not filtered_eids:
                msg = f"Claim \"{claim.claim[:60]}…\" discarded because it has no verified supporting evidence."
                logger.warning(f"[validator] {msg}")
                warnings.append(msg)
                discarded_claims_count += 1
                continue

            claim.evidence_ids = filtered_eids
            # Automatically flag inference if multiple distinct passages are cited
            if len(filtered_eids) > 1 and not claim.is_inference:
                claim.is_inference = True

            valid_claims.append(claim)

        # ── Step 3: Validate and Adjust Status ─────────────────────────────────
        final_status = response.status

        if final_status == QAStatus.ANSWERED.value:
            if not valid_evidence_items or not valid_claims:
                final_status = QAStatus.INSUFFICIENT_EVIDENCE.value
                msg = "Status changed from 'answered' to 'insufficient_evidence': no verified source evidence remained."
                warnings.append(msg)
            elif discarded_claims_count > 0:
                final_status = QAStatus.PARTIALLY_ANSWERED.value
                msg = (
                    f"Status changed from 'answered' to 'partially_answered': "
                    f"{discarded_claims_count} unverified claim(s) were discarded."
                )
                warnings.append(msg)

        elif final_status == QAStatus.PARTIALLY_ANSWERED.value:
            if not valid_evidence_items or not valid_claims:
                final_status = QAStatus.INSUFFICIENT_EVIDENCE.value
                warnings.append("Status downgraded to 'insufficient_evidence': no verified evidence remained.")

        # Ensure limitations note unverified claims if any were dropped
        final_limitations = list(response.limitations)
        if discarded_claims_count > 0:
            final_limitations.append(
                f"{discarded_claims_count} claim(s) could not be verified against the source text and were removed."
            )

        # ── Step 4: Validate Evidence Trail ───────────────────────────────────
        trail = list(response.evidence_trail)
        if not trail and valid_claims:
            # Construct a clear, verifiable evidence trail
            for idx, c in enumerate(valid_claims, 1):
                pages = []
                for eid in c.evidence_ids:
                    item = next((e for e in valid_evidence_items if e.evidence_id == eid), None)
                    if item:
                        pages.append(f"p. {item.pdf_page_index}")
                page_ref = f" (from {', '.join(pages)})" if pages else ""
                inf_tag = " [Inference]" if c.is_inference else ""
                trail.append(f"Step {idx}{inf_tag}: {c.claim}{page_ref} supported by {', '.join(c.evidence_ids)}.")

        return QAResponse(
            question=response.question,
            answer=response.answer,
            status=final_status,
            claims=valid_claims,
            evidence=valid_evidence_items,
            evidence_trail=trail,
            limitations=final_limitations,
            validation_warnings=warnings,
            metadata=dict(response.metadata),
        )
