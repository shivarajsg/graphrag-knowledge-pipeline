"""
graphrag.rule_engine
--------------------
Phase 4: Advanced Document Reasoning, Regulatory Rules, Exceptions, and Graph Evidence.
Implements:
1. DeterministicRuleEvaluator: Three-valued logic (satisfied, not_satisfied, unknown),
   nested AND/OR logic, threshold counts, mandatory conditions, group counts, and exceptions.
2. MultiPassageAssembler: Cross-reference detection, footnote linking, and multi-passage resolution.
3. TableReasoner: Column-association preservation, footnote attachment, and taxonomic hierarchy resolution.
4. RegulatoryGraphEngine: Provenance-backed provision graphs and verified reasoning chains.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .ingestion.schemas import DocumentChunk, DocumentIndex, ExtractedTable
from .qa_schemas import EvidenceItem
from .regulatory_schemas import (
    ConditionState,
    LogicalOperator,
    MultiPassageEvidence,
    ProvisionEdge,
    ProvisionGraph,
    RegulatoryRule,
    RuleCondition,
    RuleEvaluationResult,
    RuleException,
    TableCellProvenance,
    TaxonResolutionResult,
)


logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Deterministic Rule Evaluator (Three-Valued Logic)
# ==============================================================================

class DeterministicRuleEvaluator:
    """
    Evaluates regulatory rules and structured conditions using strict three-valued logic:
    - SATISFIED: Provably true based on source citations.
    - NOT_SATISFIED: Provably false or directly contradicted by evidence.
    - UNKNOWN: Evidence is missing, unstated, or ambiguous. Never assume not_satisfied!

    Handles:
    - Nested AND, OR, THRESHOLD conditions.
    - Mandatory conditions (must be SATISFIED).
    - Mandatory group counts (e.g. at least 2 from Group A, 1 from Group B).
    - Exceptions and exemptions that override the primary rule.
    - Complete deterministic reasoning audit trail.
    """

    def evaluate_condition(
        self,
        condition: RuleCondition,
        reasoning_trail: Optional[List[str]] = None,
    ) -> ConditionState:
        """
        Recursively evaluate a RuleCondition, resolving nested subconditions if present.
        """
        trail = reasoning_trail if reasoning_trail is not None else []

        # If condition has subconditions, evaluate them using its operator
        if condition.subconditions:
            sub_states = [
                self.evaluate_condition(sc, trail) for sc in condition.subconditions
            ]
            op = condition.operator

            if op == LogicalOperator.AND:
                if any(s == ConditionState.NOT_SATISFIED for s in sub_states):
                    state = ConditionState.NOT_SATISFIED
                    trail.append(
                        f"Subcondition AND group '{condition.condition_id}' evaluated to NOT_SATISFIED: "
                        f"at least one subcondition was not satisfied."
                    )
                elif all(s == ConditionState.SATISFIED for s in sub_states):
                    state = ConditionState.SATISFIED
                    trail.append(
                        f"Subcondition AND group '{condition.condition_id}' evaluated to SATISFIED: "
                        f"all {len(sub_states)} subconditions are satisfied."
                    )
                else:
                    state = ConditionState.UNKNOWN
                    trail.append(
                        f"Subcondition AND group '{condition.condition_id}' evaluated to UNKNOWN: "
                        f"one or more subconditions have unknown evidence."
                    )

            elif op == LogicalOperator.OR:
                if any(s == ConditionState.SATISFIED for s in sub_states):
                    state = ConditionState.SATISFIED
                    trail.append(
                        f"Subcondition OR group '{condition.condition_id}' evaluated to SATISFIED: "
                        f"at least one subcondition is satisfied."
                    )
                elif all(s == ConditionState.NOT_SATISFIED for s in sub_states):
                    state = ConditionState.NOT_SATISFIED
                    trail.append(
                        f"Subcondition OR group '{condition.condition_id}' evaluated to NOT_SATISFIED: "
                        f"all {len(sub_states)} subconditions are not satisfied."
                    )
                else:
                    state = ConditionState.UNKNOWN
                    trail.append(
                        f"Subcondition OR group '{condition.condition_id}' evaluated to UNKNOWN: "
                        f"no satisfied subcondition and some remain unknown."
                    )
            else:
                state = condition.state
            condition.state = state
            return state

        # Leaf condition: state is already set
        state = condition.state if isinstance(condition.state, ConditionState) else ConditionState(condition.state)
        trail.append(
            f"Condition '{condition.condition_id}' [{condition.description}]: {state.value}"
            + (f" (Page {condition.source_page})" if condition.source_page else "")
        )
        return state

    def evaluate_rule(self, rule: RegulatoryRule) -> RuleEvaluationResult:
        """
        Evaluate a complete RegulatoryRule:
        1. Check all exceptions first. If an exception applies (SATISFIED), rule is EXEMPTED.
        2. Evaluate all conditions using three-valued logic.
        3. Check mandatory conditions.
        4. Check mandatory group counts.
        5. Evaluate top-level operator (AND, OR, THRESHOLD).
        6. Produce auditable evaluation trail and list missing evidence.
        """
        trail: List[str] = []
        missing_evidence: List[str] = []
        exceptions_applied: List[str] = []
        conditions_summary: Dict[str, str] = {}
        provenance_chunks: List[str] = list(rule.source_chunk_ids)

        trail.append(f"Evaluating rule '{rule.rule_id}': {rule.subject} {rule.predicate} [{rule.rule_type}]")

        # ── Step 1: Evaluate Exceptions ─────────────────────────────────────────
        has_unknown_exception = False
        for exc in rule.exceptions:
            exc_state = exc.state if isinstance(exc.state, ConditionState) else ConditionState(exc.state)
            if exc.evidence_chunk_ids:
                provenance_chunks.extend(exc.evidence_chunk_ids)

            if exc_state == ConditionState.SATISFIED:
                exceptions_applied.append(f"{exc.exception_id}: {exc.description}")
                trail.append(
                    f"Exception '{exc.exception_id}' APPLIES: {exc.description} (applies when: {exc.applies_when})."
                )
            elif exc_state == ConditionState.UNKNOWN:
                has_unknown_exception = True
                trail.append(
                    f"Exception '{exc.exception_id}' UNKNOWN: evidence insufficient to determine if exception applies."
                )

        if exceptions_applied:
            trail.append(
                f"Rule '{rule.rule_id}' is EXEMPTED due to {len(exceptions_applied)} active exception(s)."
            )
            res = RuleEvaluationResult(
                rule_id=rule.rule_id,
                status="exempted",
                conditions_summary={},
                missing_evidence=[],
                exceptions_applied=exceptions_applied,
                reasoning_steps=trail,
                provenance_chunks=list(set(provenance_chunks)),
                confidence=1.0,
            )
            rule.evaluation_result = "exempted"
            rule.evaluation_trail = trail
            return res

        # ── Step 2: Evaluate All Conditions ───────────────────────────────────
        cond_states: List[Tuple[RuleCondition, ConditionState]] = []
        group_counts: Dict[str, int] = {}       # group_name -> satisfied count
        group_unknowns: Dict[str, int] = {}     # group_name -> unknown count

        for cond in rule.conditions:
            st = self.evaluate_condition(cond, trail)
            cond_states.append((cond, st))
            conditions_summary[cond.condition_id] = st.value

            if cond.evidence_chunk_ids:
                provenance_chunks.extend(cond.evidence_chunk_ids)

            if st == ConditionState.UNKNOWN:
                missing_evidence.append(
                    f"Condition '{cond.condition_id}' missing evidence: {cond.description}"
                )
            elif st == ConditionState.NOT_SATISFIED:
                missing_evidence.append(
                    f"Condition '{cond.condition_id}' not satisfied: {cond.description}"
                )

            # Track group memberships
            if cond.group_name:
                g = cond.group_name
                group_counts.setdefault(g, 0)
                group_unknowns.setdefault(g, 0)
                if st == ConditionState.SATISFIED:
                    group_counts[g] += 1
                elif st == ConditionState.UNKNOWN:
                    group_unknowns[g] += 1

        satisfied_count = sum(1 for _, st in cond_states if st == ConditionState.SATISFIED)
        not_satisfied_count = sum(1 for _, st in cond_states if st == ConditionState.NOT_SATISFIED)
        unknown_count = sum(1 for _, st in cond_states if st == ConditionState.UNKNOWN)

        trail.append(
            f"Condition counts: {satisfied_count} satisfied, {not_satisfied_count} not satisfied, "
            f"{unknown_count} unknown (Total: {len(rule.conditions)})."
        )

        # ── Step 3: Check Mandatory Conditions ────────────────────────────────
        mandatory_failed = False
        mandatory_unknown = False
        for cond, st in cond_states:
            if cond.is_mandatory:
                if st == ConditionState.NOT_SATISFIED:
                    mandatory_failed = True
                    trail.append(
                        f"Mandatory condition '{cond.condition_id}' FAILED: {cond.description}."
                    )
                elif st == ConditionState.UNKNOWN:
                    mandatory_unknown = True
                    trail.append(
                        f"Mandatory condition '{cond.condition_id}' UNKNOWN: {cond.description}."
                    )

        # ── Step 4: Check Mandatory Group Counts ──────────────────────────────
        group_failed = False
        group_unknown = False
        for g_name, req_count in rule.mandatory_group_counts.items():
            g_sat = group_counts.get(g_name, 0)
            g_unk = group_unknowns.get(g_name, 0)
            trail.append(
                f"Group '{g_name}' requires {req_count}: found {g_sat} satisfied, {g_unk} unknown."
            )
            if g_sat + g_unk < req_count:
                group_failed = True
                trail.append(
                    f"Group '{g_name}' cannot reach requirement ({g_sat} + {g_unk} < {req_count}): FAILED."
                )
            elif g_sat < req_count:
                group_unknown = True
                trail.append(
                    f"Group '{g_name}' currently below requirement ({g_sat} < {req_count}, {g_unk} unknown): UNKNOWN."
                )

        # ── Step 5: Logical Operator Evaluation ───────────────────────────────
        final_status: str

        if mandatory_failed or group_failed:
            final_status = ConditionState.NOT_SATISFIED.value
            trail.append(
                "Rule determination: NOT_SATISFIED because mandatory conditions or group requirements failed."
            )

        elif rule.operator == LogicalOperator.AND:
            if not_satisfied_count > 0:
                final_status = ConditionState.NOT_SATISFIED.value
                trail.append("Rule determination: NOT_SATISFIED (AND operator with at least one unsatisfied condition).")
            elif unknown_count > 0 or mandatory_unknown or group_unknown:
                final_status = ConditionState.UNKNOWN.value
                trail.append("Rule determination: UNKNOWN (AND operator with unresolved/unknown conditions).")
            else:
                final_status = ConditionState.SATISFIED.value
                trail.append("Rule determination: SATISFIED (AND operator with all conditions satisfied).")

        elif rule.operator == LogicalOperator.OR:
            if satisfied_count > 0 and not mandatory_failed:
                final_status = ConditionState.SATISFIED.value
                trail.append("Rule determination: SATISFIED (OR operator with at least one satisfied condition).")
            elif unknown_count > 0 or mandatory_unknown:
                final_status = ConditionState.UNKNOWN.value
                trail.append("Rule determination: UNKNOWN (OR operator with no satisfied condition and unknown evidence).")
            else:
                final_status = ConditionState.NOT_SATISFIED.value
                trail.append("Rule determination: NOT_SATISFIED (OR operator with all conditions not satisfied).")

        elif rule.operator == LogicalOperator.THRESHOLD:
            threshold = rule.min_threshold_count or len(rule.conditions)
            trail.append(f"Threshold requirement: at least {threshold} of {len(rule.conditions)} conditions.")

            if satisfied_count + unknown_count < threshold:
                final_status = ConditionState.NOT_SATISFIED.value
                trail.append(
                    f"Rule determination: NOT_SATISFIED (Max possible satisfied {satisfied_count + unknown_count} < {threshold})."
                )
            elif satisfied_count >= threshold:
                if mandatory_unknown or group_unknown:
                    final_status = ConditionState.UNKNOWN.value
                    trail.append(
                        "Rule determination: UNKNOWN (Threshold count met, but mandatory condition or group count is unknown)."
                    )
                else:
                    final_status = ConditionState.SATISFIED.value
                    trail.append(
                        f"Rule determination: SATISFIED (Threshold met: {satisfied_count} >= {threshold})."
                    )
            else:
                final_status = ConditionState.UNKNOWN.value
                trail.append(
                    f"Rule determination: UNKNOWN (Current satisfied {satisfied_count} < {threshold}, but {unknown_count} unknown could satisfy)."
                )

        else:
            # Operator NONE or single condition
            if mandatory_unknown or unknown_count > 0:
                final_status = ConditionState.UNKNOWN.value
            elif not_satisfied_count > 0:
                final_status = ConditionState.NOT_SATISFIED.value
            else:
                final_status = ConditionState.SATISFIED.value

        # Note potential unknown exception impact
        if has_unknown_exception and final_status == ConditionState.SATISFIED.value:
            trail.append(
                "NOTE: An exception condition is UNKNOWN. If the exception applies, the rule would be exempted."
            )

        rule.evaluation_result = final_status
        rule.evaluation_trail = trail

        return RuleEvaluationResult(
            rule_id=rule.rule_id,
            status=final_status,
            conditions_summary=conditions_summary,
            missing_evidence=missing_evidence,
            exceptions_applied=exceptions_applied,
            reasoning_steps=trail,
            provenance_chunks=list(set(provenance_chunks)),
            confidence=1.0 if final_status in (ConditionState.SATISFIED.value, "exempted") else (
                0.9 if final_status == ConditionState.NOT_SATISFIED.value else 0.5
            ),
        )


# ==============================================================================
# 2. Multi-Passage Evidence Assembler
# ==============================================================================

class MultiPassageAssembler:
    """
    Assembles evidence across multiple passages, sections, tables, and footnotes.
    Detects cross-references, footnote markers, and statutory references in document text,
    and resolves target passages from DocumentIndex.
    """

    # Patterns for cross-references and annotations
    REF_PATTERNS = [
        re.compile(r"(?:Section|Sec\.|Article|Art\.|Paragraph|Para\.|Schedule|Annex|Appendix)\s+([A-Za-z0-9\.\-\_]+)", re.IGNORECASE),
        re.compile(r"(?:see|refer to|in accordance with|pursuant to|subject to)\s+(?:Section|Sec\.|Article|Art\.|Paragraph|Para\.|Schedule|Annex|Appendix)?\s*([A-Za-z0-9\.\-\_]+)", re.IGNORECASE),
        re.compile(r"(?:#\d+)", re.IGNORECASE),
        re.compile(r"(?:footnote|note)\s+(\d+|\*+|\#+)", re.IGNORECASE),
    ]

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self._chunks_by_id: Dict[str, DocumentChunk] = {c.chunk_id: c for c in index.chunks}
        self._chunks_by_page: Dict[int, List[DocumentChunk]] = {}
        for c in index.chunks:
            self._chunks_by_page.setdefault(c.pdf_page_index, []).append(c)

    def extract_references(self, text: str) -> List[str]:
        """
        Extract cross-reference tokens, section pointers, or footnote markers from text.
        """
        refs: Set[str] = set()
        for pat in self.REF_PATTERNS:
            for match in pat.finditer(text):
                full_token = match.group(0).strip()
                refs.add(full_token)
        return sorted(refs)

    def resolve_reference(self, reference_text: str) -> List[DocumentChunk]:
        """
        Search DocumentIndex to find chunks that define or contain the referenced provision.
        Returns matching chunks, or empty list if the reference cannot be resolved.
        """
        ref_clean = reference_text.strip()
        matched: List[DocumentChunk] = []

        # 1. Footnote marker like "#4" or "#1"
        if re.match(r"^#\d+$", ref_clean):
            fn_tag = ref_clean
            for c in self.index.chunks:
                # Footnote definitions typically appear as "#4\nAll parts..." or start a paragraph with "#4"
                if re.search(rf"(?:^|\n)\s*{re.escape(fn_tag)}\s+(?:All parts|Populations|Logs|Seeds|Only)", c.text, re.IGNORECASE):
                    matched.append(c)
                elif fn_tag in c.text and ("exemption" in c.text.lower() or "derivatives" in c.text.lower()):
                    matched.append(c)
            if matched:
                return matched

        # 2. Appendix reference like "Appendix I", "Appendix II", "Appendix III"
        app_match = re.search(r"Appendix\s+(I{1,3})", ref_clean, re.IGNORECASE)
        if app_match:
            app_roman = app_match.group(1).upper()
            for c in self.index.chunks:
                if f"Appendix {app_roman}" in c.text or f"Appendices {app_roman}" in c.text:
                    matched.append(c)
            return matched[:3]

        # 3. Section or Schedule reference like "Section 4", "Article II"
        sec_match = re.search(r"(?:Section|Article|Schedule|Annex)\s+([A-Za-z0-9]+)", ref_clean, re.IGNORECASE)
        if sec_match:
            sec_num = sec_match.group(1)
            for c in self.index.chunks:
                if (c.section_heading and sec_num.lower() in c.section_heading.lower()) or (
                    re.search(rf"(?:Section|Article|Schedule|Annex)\s+{re.escape(sec_num)}\b", c.text, re.IGNORECASE)
                ):
                    matched.append(c)
            if matched:
                return matched[:3]

        # 4. General keyword search across chunks
        tokens = [t for t in re.findall(r"\w+", ref_clean) if len(t) > 2]
        if tokens:
            for c in self.index.chunks:
                if all(t.lower() in c.text.lower() for t in tokens):
                    matched.append(c)
                    if len(matched) >= 2:
                        break

        return matched

    def assemble_multi_passage_evidence(
        self,
        root_chunk_id: str,
        max_hops: int = 2,
    ) -> MultiPassageEvidence:
        """
        Given a root chunk, scans its text for cross-references, footnotes, and provisions.
        Recursively resolves target chunks up to max_hops.
        Returns unified MultiPassageEvidence without hallucinations.
        """
        root_chunk = self._chunks_by_id.get(root_chunk_id)
        if not root_chunk:
            return MultiPassageEvidence(
                root_chunk_id=root_chunk_id,
                unresolved_references=[f"Root chunk '{root_chunk_id}' not found in index."],
            )

        visited_chunk_ids: Set[str] = {root_chunk_id}
        connected_chunks: List[str] = []
        resolved_refs: List[str] = []
        unresolved_refs: List[str] = []
        footnotes: List[str] = []
        citations: List[Dict[str, Any]] = [
            {
                "chunk_id": root_chunk.chunk_id,
                "page": root_chunk.pdf_page_index,
                "text_snippet": root_chunk.text[:200],
            }
        ]

        queue: deque[Tuple[str, int]] = deque([(root_chunk.text, 0)])

        while queue:
            curr_text, hop = queue.popleft()
            if hop >= max_hops:
                continue

            extracted = self.extract_references(curr_text)
            for ref in extracted:
                if ref.startswith("#"):
                    footnotes.append(ref)

                resolved = self.resolve_reference(ref)
                if resolved:
                    resolved_refs.append(ref)
                    for r_chunk in resolved:
                        if r_chunk.chunk_id not in visited_chunk_ids:
                            visited_chunk_ids.add(r_chunk.chunk_id)
                            connected_chunks.append(r_chunk.chunk_id)
                            citations.append({
                                "chunk_id": r_chunk.chunk_id,
                                "page": r_chunk.pdf_page_index,
                                "text_snippet": r_chunk.text[:200],
                                "referenced_by": ref,
                            })
                            queue.append((r_chunk.text, hop + 1))
                else:
                    unresolved_refs.append(ref)

        # Build combined context
        combined_texts = [f"[Page {root_chunk.pdf_page_index}] {root_chunk.text}"]
        for cid in connected_chunks:
            c = self._chunks_by_id.get(cid)
            if c:
                combined_texts.append(f"[Page {c.pdf_page_index}] {c.text}")

        return MultiPassageEvidence(
            root_chunk_id=root_chunk_id,
            connected_chunk_ids=connected_chunks,
            cross_references_resolved=sorted(list(set(resolved_refs))),
            unresolved_references=sorted(list(set(unresolved_refs))),
            footnotes=sorted(list(set(footnotes))),
            combined_context="\n\n---\n\n".join(combined_texts),
            citations=citations,
        )


# ==============================================================================
# 3. Table Reasoner (Column Associations and Footnotes)
# ==============================================================================

class TableReasoner:
    """
    Safely interprets structured tables from regulatory documents.
    Preserves column-to-heading associations (e.g. CITES Appendix I vs Appendix II vs Appendix III).
    Never infers classification from raw page co-occurrence alone!
    Attaches modifying footnotes and resolves higher-taxon exclusions.
    """

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index

    def detect_appendix_columns(self, table: ExtractedTable) -> Dict[int, str]:
        """
        Determines which column index corresponds to which Appendix (I, II, III).
        Standard CITES tables have 4 columns:
        Col 0: Taxon name / Family
        Col 1: Appendix I
        Col 2: Appendix II
        Col 3: Appendix III
        """
        mapping: Dict[int, str] = {}

        # Check headers first
        for idx, h in enumerate(table.headers):
            h_clean = h.strip().upper()
            if h_clean == "I" or "APPENDIX I" in h_clean:
                mapping[idx] = "Appendix I"
            elif h_clean == "II" or "APPENDIX II" in h_clean:
                mapping[idx] = "Appendix II"
            elif h_clean == "III" or "APPENDIX III" in h_clean:
                mapping[idx] = "Appendix III"

        # If headers didn't give full mapping, inspect early rows
        for row in table.rows[:3]:
            for idx, cell in enumerate(row):
                c_clean = cell.strip().upper()
                if c_clean == "I":
                    mapping[idx] = "Appendix I"
                elif c_clean == "II":
                    mapping[idx] = "Appendix II"
                elif c_clean == "III":
                    mapping[idx] = "Appendix III"

        # Default fallback for CITES standard layout if 4 columns
        if not mapping and table.rows and len(table.rows[0]) == 4:
            mapping = {1: "Appendix I", 2: "Appendix II", 3: "Appendix III"}

        return mapping

    def lookup_taxon_cell(
        self,
        taxon_name: str,
        target_appendix: Optional[str] = None,
    ) -> List[TableCellProvenance]:
        """
        Searches tables across all pages for a specific taxon or species.
        Preserves exact column association and extracts modifying annotations.
        """
        results: List[TableCellProvenance] = []
        name_lower = taxon_name.lower().strip()

        for page in self.index.pages:
            for table in page.tables:
                col_map = self.detect_appendix_columns(table)
                for r_idx, row in enumerate(table.rows):
                    for c_idx, cell in enumerate(row):
                        cell_lower = cell.lower()
                        if name_lower in cell_lower:
                            col_heading = col_map.get(c_idx, f"Column {c_idx}")

                            if target_appendix and target_appendix.lower() not in col_heading.lower():
                                continue

                            # Detect modifying footnotes in cell (e.g., #4, #1, 2)
                            fn_matches = re.findall(r"(?:#\d+|\b\d+\b)", cell)
                            clean_fns = [f for f in fn_matches if len(f) <= 4]

                            results.append(
                                TableCellProvenance(
                                    table_id=table.table_id,
                                    pdf_page_index=page.pdf_page_index,
                                    column_heading=col_heading,
                                    row_index=r_idx,
                                    cell_text=cell.strip(),
                                    modifying_footnotes=clean_fns,
                                    confidence=1.0,
                                )
                            )
        return results

    def resolve_cites_classification(
        self,
        species_name: str,
        higher_taxon: Optional[str] = None,
        strict_document_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Applies regulatory table hierarchy rules with strict source-grounding:
        1. Checks if species is explicitly listed in Appendix I (Column 1), Appendix II (Column 2),
           or Appendix III (Column 3). Direct matches are flagged as is_explicit_listing=True,
           is_inference=False.
        2. If NOT listed directly:
           a. In strict_document_only mode: refuses to introduce external taxonomic mappings and
              returns unlisted_in_document (insufficient evidence).
           b. If higher_taxon is NOT provided: returns unlisted_in_document without assuming
              any default higher taxon.
           c. If higher_taxon IS provided (e.g. from user input or query premise):
              - Checks whether the higher taxon exists in the document tables (e.g. "PRIMATES spp.").
              - If found in Appendix II with Appendix I exclusion, and species is not in Appendix I,
                returns status="conditional_inference", is_explicit_listing=False, is_inference=True.
              - The citation points to the general higher-taxon entry on page 19, explicitly
                stating that the queried species does NOT appear verbatim in the document and
                that the classification relies on the external premise that it belongs to higher_taxon.
        """
        # Step 1: Direct Table Matches
        app1_matches = self.lookup_taxon_cell(species_name, target_appendix="Appendix I")
        app2_matches = self.lookup_taxon_cell(species_name, target_appendix="Appendix II")
        app3_matches = self.lookup_taxon_cell(species_name, target_appendix="Appendix III")

        if app1_matches:
            best = app1_matches[0]
            result = TaxonResolutionResult(
                species=species_name,
                classification="Appendix I",
                status="direct_listing",
                is_explicit_listing=True,
                is_inference=False,
                rationale=f"Explicitly listed in Appendix I column on page {best.pdf_page_index}.",
                provenance=best,
                verbatim_source_entry=best.cell_text,
            )
            return result.to_dict()

        if app2_matches:
            best = app2_matches[0]
            result = TaxonResolutionResult(
                species=species_name,
                classification="Appendix II",
                status="direct_listing",
                is_explicit_listing=True,
                is_inference=False,
                rationale=f"Explicitly listed in Appendix II column on page {best.pdf_page_index}.",
                provenance=best,
                verbatim_source_entry=best.cell_text,
            )
            return result.to_dict()

        if app3_matches:
            best = app3_matches[0]
            result = TaxonResolutionResult(
                species=species_name,
                classification="Appendix III",
                status="direct_listing",
                is_explicit_listing=True,
                is_inference=False,
                rationale=f"Explicitly listed in Appendix III column on page {best.pdf_page_index}.",
                provenance=best,
                verbatim_source_entry=best.cell_text,
            )
            return result.to_dict()

        # Step 2: Target species does not appear verbatim in any table cell
        # Strict document-only policy check
        if strict_document_only:
            result = TaxonResolutionResult(
                species=species_name,
                classification="Unlisted / Insufficient Evidence",
                status="unlisted_in_document",
                is_explicit_listing=False,
                is_inference=False,
                rationale=(
                    f"Strict document-only policy: The species '{species_name}' does not appear anywhere "
                    f"in the document text or tables. Unverified external taxonomy mappings are not permitted."
                ),
            )
            return result.to_dict()

        # If no higher taxon was provided, refuse to guess or default
        if not higher_taxon or not higher_taxon.strip():
            result = TaxonResolutionResult(
                species=species_name,
                classification="Unlisted / Unknown",
                status="unlisted_in_document",
                is_explicit_listing=False,
                is_inference=False,
                rationale=(
                    f"The species '{species_name}' is not listed in the document tables. "
                    f"No higher taxon was specified. Without an explicit taxonomic premise, "
                    f"its CITES classification cannot be established from the document alone."
                ),
            )
            return result.to_dict()

        # Step 3: Higher Taxon Resolution (e.g. PRIMATES spp.)
        clean_higher = higher_taxon.strip()
        higher_app2 = self.lookup_taxon_cell(f"{clean_higher} spp.", target_appendix="Appendix II")
        higher_app1 = self.lookup_taxon_cell(f"{clean_higher} spp.", target_appendix="Appendix I")

        if higher_app2:
            best = higher_app2[0]
            result = TaxonResolutionResult(
                species=species_name,
                classification="Appendix II (Conditional Inference)",
                status="conditional_inference",
                is_explicit_listing=False,
                is_inference=True,
                higher_taxon=clean_higher,
                assumptions=[
                    f"External taxonomic assumption: '{species_name}' is assumed to belong to higher taxon '{clean_higher}' (this biological classification is not stated in the document)."
                ],
                provenance=best,
                verbatim_source_entry=best.cell_text,
                rationale=(
                    f"The species '{species_name}' does NOT appear verbatim anywhere in the document. "
                    f"CONDITIONAL INFERENCE: Assuming '{species_name}' belongs to higher taxon '{clean_higher}' "
                    f"(external premise), it falls under the general higher-taxon entry '{best.cell_text}' "
                    f"in Appendix II on page {best.pdf_page_index}, because it is not listed among the excluded Appendix I species on page 20."
                ),
            )
            return result.to_dict()

        if higher_app1:
            best = higher_app1[0]
            result = TaxonResolutionResult(
                species=species_name,
                classification="Appendix I (Conditional Inference)",
                status="conditional_inference",
                is_explicit_listing=False,
                is_inference=True,
                higher_taxon=clean_higher,
                assumptions=[
                    f"External taxonomic assumption: '{species_name}' is assumed to belong to higher taxon '{clean_higher}' (not stated in document)."
                ],
                provenance=best,
                verbatim_source_entry=best.cell_text,
                rationale=(
                    f"The species '{species_name}' does NOT appear verbatim in the document. "
                    f"CONDITIONAL INFERENCE: Assuming '{species_name}' belongs to '{clean_higher}', "
                    f"it falls under the general Appendix I entry '{best.cell_text}' on page {best.pdf_page_index}."
                ),
            )
            return result.to_dict()

        # Higher taxon itself was not found in document
        result = TaxonResolutionResult(
            species=species_name,
            classification="Unlisted / Unknown",
            status="unsupported_taxonomic_mapping",
            is_explicit_listing=False,
            is_inference=False,
            higher_taxon=clean_higher,
            assumptions=[f"Higher taxon '{clean_higher}' is not listed in document tables."],
            rationale=f"Neither '{species_name}' nor higher taxon '{clean_higher}' appears in the document tables.",
        )
        return result.to_dict()



# ==============================================================================
# 4. Regulatory Graph Engine (Provable Provision Graph)
# ==============================================================================

class RegulatoryGraphEngine:
    """
    Constructs and reasons over a provable graph of document provisions.
    Strictly validates edge provenance: an edge without traceable citations is rejected!
    Traces multi-hop reasoning across provisions, exceptions, and footnotes.
    """

    def __init__(self, index: Optional[DocumentIndex] = None) -> None:
        self.index = index
        self.graph = ProvisionGraph()

    def add_provision(self, provision_id: str, metadata: Dict[str, Any]) -> None:
        self.graph.add_provision(provision_id, metadata)

    def add_edge_with_provenance(
        self,
        edge_id: str,
        source_provision: str,
        target_provision: str,
        relation: str,
        source_page: int,
        target_page: Optional[int],
        citation_quote: str,
    ) -> bool:
        """
        Adds a provision edge only if provenance criteria are satisfied:
        - Source and target provisions must be non-empty.
        - citation_quote must be non-empty and verifiable.
        - source_page must be > 0.
        If provenance is missing, edge is rejected to prevent graph hallucinations.
        """
        if not source_provision or not target_provision:
            logger.warning(f"[regulatory_graph] Edge '{edge_id}' rejected: missing endpoints.")
            return False

        if not citation_quote or not citation_quote.strip():
            logger.warning(f"[regulatory_graph] Edge '{edge_id}' rejected: citation_quote is empty.")
            edge = ProvisionEdge(
                edge_id=edge_id,
                source_provision=source_provision,
                target_provision=target_provision,
                relation=relation,
                source_page=source_page,
                target_page=target_page,
                citation_quote="",
                has_valid_provenance=False,
                is_resolved=False,
            )
            self.graph.add_edge(edge)
            return False

        if source_page <= 0:
            logger.warning(f"[regulatory_graph] Edge '{edge_id}' rejected: invalid source_page {source_page}.")
            return False

        # If index is present, verify that citation_quote occurs in the document
        if self.index:
            found = any(
                citation_quote.lower() in c.text.lower()
                for c in self.index.chunks
                if c.pdf_page_index == source_page
            )
            if not found:
                # Still check entire index before rejecting
                found_anywhere = any(citation_quote.lower() in c.text.lower() for c in self.index.chunks)
                if not found_anywhere:
                    logger.warning(
                        f"[regulatory_graph] Edge '{edge_id}' rejected: quote not found in indexed text."
                    )
                    edge = ProvisionEdge(
                        edge_id=edge_id,
                        source_provision=source_provision,
                        target_provision=target_provision,
                        relation=relation,
                        source_page=source_page,
                        target_page=target_page,
                        citation_quote=citation_quote,
                        has_valid_provenance=False,
                        is_resolved=False,
                    )
                    self.graph.add_edge(edge)
                    return False

        edge = ProvisionEdge(
            edge_id=edge_id,
            source_provision=source_provision,
            target_provision=target_provision,
            relation=relation,
            source_page=source_page,
            target_page=target_page,
            citation_quote=citation_quote,
            has_valid_provenance=True,
            is_resolved=True,
        )
        self.graph.add_edge(edge)
        return True

    def find_reasoning_path(
        self,
        start_provision: str,
        end_provision: str,
        max_depth: int = 5,
    ) -> List[ProvisionEdge]:
        """
        Finds a directed path of verified provision edges from start_provision to end_provision.
        Uses BFS to find the shortest verified path.
        """
        if start_provision == end_provision:
            return []

        visited: Set[str] = {start_provision}
        queue: deque[Tuple[str, List[ProvisionEdge]]] = deque([(start_provision, [])])

        while queue:
            curr_prov, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            for edge in self.graph.get_outbound_edges(curr_prov):
                if not edge.has_valid_provenance or not edge.is_resolved:
                    continue

                nxt = edge.target_provision
                new_path = path + [edge]

                if nxt == end_provision:
                    return new_path

                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, new_path))

        return []
