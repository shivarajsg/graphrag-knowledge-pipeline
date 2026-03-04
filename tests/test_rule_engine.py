"""
tests.test_rule_engine
----------------------
Unit and integration tests for Phase 4:
- Three-valued logic (SATISFIED, NOT_SATISFIED, UNKNOWN)
- Mandatory conditions and threshold counts
- Nested AND/OR conditions
- Exceptions and exemptions
- Multi-passage cross-reference resolution and unresolved reference safety
- Table reasoner column-association preservation (CITES Appendix I vs II)
- Footnote extraction and attachment
- Provable provision graph with edge provenance validation
- Real-world evaluation against CITES index
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from graphrag.ingestion.schemas import (
    DocumentChunk,
    DocumentIndex,
    ExtractedPage,
    ExtractedTable,
)
from graphrag.qa_engine import DocumentQAEngine, MockQAClient
from graphrag.qa_schemas import QAStatus
from graphrag.regulatory_schemas import (
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
)
from graphrag.rule_engine import (
    DeterministicRuleEvaluator,
    MultiPassageAssembler,
    RegulatoryGraphEngine,
    TableReasoner,
)

CITES_INDEX_PATH = Path("outputs/cites_index.json")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator() -> DeterministicRuleEvaluator:
    return DeterministicRuleEvaluator()


@pytest.fixture
def cites_index() -> DocumentIndex:
    if not CITES_INDEX_PATH.exists():
        pytest.skip(f"CITES index not found at {CITES_INDEX_PATH}")
    return DocumentIndex.load(CITES_INDEX_PATH)


@pytest.fixture
def mock_regulatory_index() -> DocumentIndex:
    """A synthetic 3-page index with structured provisions, cross-references, and tables."""
    page1 = ExtractedPage(
        pdf_page_index=1,
        printed_page_number="1",
        text="Section 1: General Export Controls. All commercial wildlife shipments require valid permits under Section 4.",
    )
    table1 = ExtractedTable(
        table_id="tbl_p2_0",
        headers=["Taxon", "Appendix I", "Appendix II", "Appendix III"],
        rows=[
            ["", "I", "II", "III"],
            ["Felidae spp.", "Panthera leo", "Felis catus", ""],
            ["Ursidae spp.", "Ursus arctos #1", "Ursus maritimus", ""],
        ],
    )
    page2 = ExtractedPage(
        pdf_page_index=2,
        printed_page_number="2",
        text="Section 2: Classification Table. Refer to Table 1 for species listings.",
        tables=[table1],
    )
    page3 = ExtractedPage(
        pdf_page_index=3,
        printed_page_number="3",
        text="Section 4: Permit Issuance Requirements.\n#1 Valid only for captive-bred populations.\nExcept that scientific exchanges between registered institutions are exempt from export permits.",
    )

    chunks = [
        DocumentChunk(
            chunk_id="chunk_p1_c0",
            document_id="doc_reg_01",
            source_file="regulatory_sample.pdf",
            document_hash="abc123hash",
            pdf_page_index=1,
            printed_page_number="1",
            chunk_index=0,
            text=page1.text,
            page_char_start=0,
            page_char_end=len(page1.text),
        ),
        DocumentChunk(
            chunk_id="chunk_p2_c0",
            document_id="doc_reg_01",
            source_file="regulatory_sample.pdf",
            document_hash="abc123hash",
            pdf_page_index=2,
            printed_page_number="2",
            chunk_index=0,
            text=page2.text,
            page_char_start=0,
            page_char_end=len(page2.text),
        ),
        DocumentChunk(
            chunk_id="chunk_p3_c0",
            document_id="doc_reg_01",
            source_file="regulatory_sample.pdf",
            document_hash="abc123hash",
            pdf_page_index=3,
            printed_page_number="3",
            chunk_index=0,
            text=page3.text,
            page_char_start=0,
            page_char_end=len(page3.text),
        ),

    ]

    return DocumentIndex(
        document_id="doc_reg_01",
        source_file="regulatory_sample.pdf",
        document_hash="abc123hash",
        total_pages=3,
        total_chunks=3,
        pages=[page1, page2, page3],
        chunks=chunks,
    )


# ==============================================================================
# Scenario 1: Three-Valued Logic (Kleene Logic)
# ==============================================================================

def test_three_valued_logic_kleene_and(evaluator: DeterministicRuleEvaluator):
    """Test Kleene 3-valued AND: SATISFIED, NOT_SATISFIED, and UNKNOWN."""
    # (SATISFIED, SATISFIED) -> SATISFIED
    rule_sat = RegulatoryRule(
        rule_id="R_AND_SAT",
        rule_type="requirement",
        subject="Exporter",
        predicate="must satisfy criteria",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("C1", "Has license", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Paid fee", state=ConditionState.SATISFIED),
        ],
    )
    res_sat = evaluator.evaluate_rule(rule_sat)
    assert res_sat.status == "satisfied"

    # (SATISFIED, UNKNOWN) -> UNKNOWN
    rule_unk = RegulatoryRule(
        rule_id="R_AND_UNK",
        rule_type="requirement",
        subject="Exporter",
        predicate="must satisfy criteria",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("C1", "Has license", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Inspection verified", state=ConditionState.UNKNOWN),
        ],
    )
    res_unk = evaluator.evaluate_rule(rule_unk)
    assert res_unk.status == "unknown"
    assert any("Inspection verified" in m for m in res_unk.missing_evidence)

    # (SATISFIED, NOT_SATISFIED) -> NOT_SATISFIED
    rule_not = RegulatoryRule(
        rule_id="R_AND_NOT",
        rule_type="requirement",
        subject="Exporter",
        predicate="must satisfy criteria",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("C1", "Has license", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Banned substance check", state=ConditionState.NOT_SATISFIED),
        ],
    )
    res_not = evaluator.evaluate_rule(rule_not)
    assert res_not.status == "not_satisfied"


def test_three_valued_logic_kleene_or(evaluator: DeterministicRuleEvaluator):
    """Test Kleene 3-valued OR: SATISFIED overrides UNKNOWN; UNKNOWN overrides NOT_SATISFIED."""
    # (NOT_SATISFIED, SATISFIED) -> SATISFIED
    rule_sat = RegulatoryRule(
        rule_id="R_OR_SAT",
        rule_type="permission",
        subject="Exporter",
        predicate="may qualify via alternate methods",
        operator=LogicalOperator.OR,
        conditions=[
            RuleCondition("C1", "Method A", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C2", "Method B", state=ConditionState.SATISFIED),
        ],
    )
    assert evaluator.evaluate_rule(rule_sat).status == "satisfied"

    # (NOT_SATISFIED, UNKNOWN) -> UNKNOWN
    rule_unk = RegulatoryRule(
        rule_id="R_OR_UNK",
        rule_type="permission",
        subject="Exporter",
        predicate="may qualify via alternate methods",
        operator=LogicalOperator.OR,
        conditions=[
            RuleCondition("C1", "Method A", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C2", "Method B", state=ConditionState.UNKNOWN),
        ],
    )
    assert evaluator.evaluate_rule(rule_unk).status == "unknown"

    # (NOT_SATISFIED, NOT_SATISFIED) -> NOT_SATISFIED
    rule_not = RegulatoryRule(
        rule_id="R_OR_NOT",
        rule_type="permission",
        subject="Exporter",
        predicate="may qualify via alternate methods",
        operator=LogicalOperator.OR,
        conditions=[
            RuleCondition("C1", "Method A", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C2", "Method B", state=ConditionState.NOT_SATISFIED),
        ],
    )
    assert evaluator.evaluate_rule(rule_not).status == "not_satisfied"


# ==============================================================================
# Scenario 2: Mandatory Condition Failure
# ==============================================================================

def test_mandatory_condition_failure(evaluator: DeterministicRuleEvaluator):
    """Even if 5 of 6 conditions pass, failure of 1 mandatory condition fails the entire rule."""
    rule = RegulatoryRule(
        rule_id="R_MANDATORY_FAIL",
        rule_type="requirement",
        subject="Applicant",
        predicate="qualifies for certificate",
        operator=LogicalOperator.THRESHOLD,
        min_threshold_count=4,
        conditions=[
            RuleCondition("C1", "Valid passport", is_mandatory=True, state=ConditionState.NOT_SATISFIED),
            RuleCondition("C2", "Background check", is_mandatory=False, state=ConditionState.SATISFIED),
            RuleCondition("C3", "Financial solvency", is_mandatory=False, state=ConditionState.SATISFIED),
            RuleCondition("C4", "Local residence", is_mandatory=False, state=ConditionState.SATISFIED),
            RuleCondition("C5", "Health insurance", is_mandatory=False, state=ConditionState.SATISFIED),
            RuleCondition("C6", "Application fee", is_mandatory=False, state=ConditionState.SATISFIED),
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == "not_satisfied"
    assert any("Valid passport" in step for step in res.reasoning_steps)


def test_mandatory_condition_unknown(evaluator: DeterministicRuleEvaluator):
    """If a mandatory condition is UNKNOWN, the rule cannot evaluate to satisfied."""
    rule = RegulatoryRule(
        rule_id="R_MANDATORY_UNK",
        rule_type="requirement",
        subject="Applicant",
        predicate="qualifies for certificate",
        operator=LogicalOperator.THRESHOLD,
        min_threshold_count=3,
        conditions=[
            RuleCondition("C1", "Valid passport", is_mandatory=True, state=ConditionState.UNKNOWN),
            RuleCondition("C2", "Background check", is_mandatory=False, state=ConditionState.SATISFIED),
            RuleCondition("C3", "Financial solvency", is_mandatory=False, state=ConditionState.SATISFIED),
            RuleCondition("C4", "Local residence", is_mandatory=False, state=ConditionState.SATISFIED),
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == "unknown"


# ==============================================================================
# Scenario 3: Threshold Count Evaluation (e.g. 4 of 6)
# ==============================================================================

def test_threshold_count_evaluation(evaluator: DeterministicRuleEvaluator):
    """Verify threshold evaluation (e.g. 4 of 6 conditions satisfied)."""
    # 4 of 6 satisfied -> satisfied
    rule_pass = RegulatoryRule(
        rule_id="R_THRESH_PASS",
        rule_type="requirement",
        subject="Vessel",
        predicate="meets safety standards",
        operator=LogicalOperator.THRESHOLD,
        min_threshold_count=4,
        conditions=[
            RuleCondition("C1", "Life rafts", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Flares", state=ConditionState.SATISFIED),
            RuleCondition("C3", "Radio", state=ConditionState.SATISFIED),
            RuleCondition("C4", "Fire extinguishers", state=ConditionState.SATISFIED),
            RuleCondition("C5", "Radar", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C6", "Sonar", state=ConditionState.NOT_SATISFIED),
        ],
    )
    assert evaluator.evaluate_rule(rule_pass).status == "satisfied"

    # 3 satisfied, 1 unknown, 2 not satisfied -> unknown (3 + 1 = 4 is still reachable)
    rule_unk = RegulatoryRule(
        rule_id="R_THRESH_UNK",
        rule_type="requirement",
        subject="Vessel",
        predicate="meets safety standards",
        operator=LogicalOperator.THRESHOLD,
        min_threshold_count=4,
        conditions=[
            RuleCondition("C1", "Life rafts", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Flares", state=ConditionState.SATISFIED),
            RuleCondition("C3", "Radio", state=ConditionState.SATISFIED),
            RuleCondition("C4", "Fire extinguishers", state=ConditionState.UNKNOWN),
            RuleCondition("C5", "Radar", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C6", "Sonar", state=ConditionState.NOT_SATISFIED),
        ],
    )
    assert evaluator.evaluate_rule(rule_unk).status == "unknown"

    # 2 satisfied, 1 unknown, 3 not satisfied -> not_satisfied (max possible 3 < 4)
    rule_fail = RegulatoryRule(
        rule_id="R_THRESH_FAIL",
        rule_type="requirement",
        subject="Vessel",
        predicate="meets safety standards",
        operator=LogicalOperator.THRESHOLD,
        min_threshold_count=4,
        conditions=[
            RuleCondition("C1", "Life rafts", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Flares", state=ConditionState.SATISFIED),
            RuleCondition("C3", "Radio", state=ConditionState.UNKNOWN),
            RuleCondition("C4", "Fire extinguishers", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C5", "Radar", state=ConditionState.NOT_SATISFIED),
            RuleCondition("C6", "Sonar", state=ConditionState.NOT_SATISFIED),
        ],
    )
    assert evaluator.evaluate_rule(rule_fail).status == "not_satisfied"


# ==============================================================================
# Scenario 4: Mandatory Group Counts
# ==============================================================================

def test_mandatory_group_counts(evaluator: DeterministicRuleEvaluator):
    """Rule requiring at least 2 conditions from Group A and 1 from Group B."""
    rule = RegulatoryRule(
        rule_id="R_GROUPS",
        rule_type="requirement",
        subject="Facility",
        predicate="meets environmental standards",
        operator=LogicalOperator.THRESHOLD,
        min_threshold_count=3,
        mandatory_group_counts={"Air": 2, "Water": 1},
        conditions=[
            RuleCondition("A1", "Air filter A", group_name="Air", state=ConditionState.SATISFIED),
            RuleCondition("A2", "Air filter B", group_name="Air", state=ConditionState.NOT_SATISFIED),
            RuleCondition("A3", "Air stack test", group_name="Air", state=ConditionState.NOT_SATISFIED),
            RuleCondition("W1", "Water treatment", group_name="Water", state=ConditionState.SATISFIED),
            RuleCondition("W2", "Water monitoring", group_name="Water", state=ConditionState.SATISFIED),
        ],
    )
    # Total satisfied is 3 (A1, W1, W2), but Group Air has only 1 satisfied and 0 unknown -> Air requirement (2) fails!
    res = evaluator.evaluate_rule(rule)
    assert res.status == "not_satisfied"
    assert any("Air" in step for step in res.reasoning_steps)


# ==============================================================================
# Scenario 5: Nested AND/OR Conditions
# ==============================================================================

def test_nested_and_or_conditions(evaluator: DeterministicRuleEvaluator):
    """Test nested condition trees: Cond1 AND (Cond2 OR Cond3)."""
    sub_or = RuleCondition(
        condition_id="C_SUB_OR",
        description="Either bond or bank guarantee",
        operator=LogicalOperator.OR,
        subconditions=[
            RuleCondition("B1", "Cash bond", state=ConditionState.NOT_SATISFIED),
            RuleCondition("B2", "Bank guarantee", state=ConditionState.SATISFIED),
        ],
    )
    rule = RegulatoryRule(
        rule_id="R_NESTED",
        rule_type="requirement",
        subject="Operator",
        predicate="authorized",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("A1", "Operating license", state=ConditionState.SATISFIED),
            sub_or,
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == "satisfied"
    assert sub_or.state == ConditionState.SATISFIED


# ==============================================================================
# Scenario 6: Rule Exception Override (Exemptions)
# ==============================================================================

def test_rule_exception_override(evaluator: DeterministicRuleEvaluator):
    """If an exception condition applies (SATISFIED), the rule status becomes 'exempted'."""
    rule = RegulatoryRule(
        rule_id="R_PROHIBITION_WITH_EXCEPTION",
        rule_type="prohibition",
        subject="Specimen",
        predicate="commercial trade prohibited",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("P1", "Species in Appendix I", state=ConditionState.SATISFIED),
        ],
        exceptions=[
            RuleException(
                exception_id="EXC_1",
                description="Artificially propagated plant exemption",
                applies_when="Specimen is grown under controlled artificial conditions",
                state=ConditionState.SATISFIED,
                evidence_chunk_ids=["chunk_p3_c0"],
                source_page=3,
            )
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == "exempted"
    assert len(res.exceptions_applied) == 1
    assert "EXC_1" in res.exceptions_applied[0]


def test_rule_exception_not_applied(evaluator: DeterministicRuleEvaluator):
    """If exception condition does not apply (NOT_SATISFIED), primary rule evaluation stands."""
    rule = RegulatoryRule(
        rule_id="R_PROHIBITION_STANDS",
        rule_type="prohibition",
        subject="Specimen",
        predicate="commercial trade prohibited",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("P1", "Species in Appendix I", state=ConditionState.SATISFIED),
        ],
        exceptions=[
            RuleException(
                exception_id="EXC_1",
                description="Artificially propagated plant exemption",
                applies_when="Specimen is artificially propagated",
                state=ConditionState.NOT_SATISFIED,
            )
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == "satisfied"
    assert not res.exceptions_applied


# ==============================================================================
# Scenario 7: Missing Evidence Never Converted to False
# ==============================================================================

def test_unknown_never_assumed_false(evaluator: DeterministicRuleEvaluator):
    """Unknown condition with no evidence must remain UNKNOWN and listed in missing_evidence."""
    rule = RegulatoryRule(
        rule_id="R_UNKNOWN_SAFEGUARD",
        rule_type="requirement",
        subject="Cargo",
        predicate="cleared for transit",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition("C1", "Quarantine certificate", state=ConditionState.SATISFIED),
            RuleCondition("C2", "Host country verification", state=ConditionState.UNKNOWN),
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == ConditionState.UNKNOWN.value
    assert len(res.missing_evidence) == 1
    assert "Host country verification" in res.missing_evidence[0]


# ==============================================================================
# Scenario 8: Provision Graph Edge Provenance Validation
# ==============================================================================

def test_provision_graph_edge_provenance_validation(mock_regulatory_index: DocumentIndex):
    """Edges lacking valid provenance or exact citations must be rejected by the graph."""
    engine = RegulatoryGraphEngine(index=mock_regulatory_index)
    engine.add_provision("Section 1", {"title": "General Controls"})
    engine.add_provision("Section 4", {"title": "Permit Requirements"})

    # Case 1: Empty citation quote -> Rejected!
    success1 = engine.add_edge_with_provenance(
        edge_id="edge_1_to_4_invalid",
        source_provision="Section 1",
        target_provision="Section 4",
        relation="references",
        source_page=1,
        target_page=3,
        citation_quote="",  # Missing quote
    )
    assert success1 is False
    assert len(engine.graph.edges) == 0

    # Case 2: Quote not present in document index -> Rejected!
    success2 = engine.add_edge_with_provenance(
        edge_id="edge_hallucinated",
        source_provision="Section 1",
        target_provision="Section 4",
        relation="references",
        source_page=1,
        target_page=3,
        citation_quote="This sentence never appeared in the text anywhere at all.",
    )
    assert success2 is False
    assert len(engine.graph.edges) == 0

    # Case 3: Valid verifiable citation quote -> Accepted!
    success3 = engine.add_edge_with_provenance(
        edge_id="edge_valid",
        source_provision="Section 1",
        target_provision="Section 4",
        relation="references",
        source_page=1,
        target_page=3,
        citation_quote="All commercial wildlife shipments require valid permits under Section 4.",
    )
    assert success3 is True
    assert len(engine.graph.edges) == 1
    assert engine.graph.edges[0].edge_id == "edge_valid"
    assert engine.graph.edges[0].has_valid_provenance is True


# ==============================================================================
# Scenario 9: Multi-Hop Reasoning Path
# ==============================================================================

def test_provision_graph_multi_hop_chain():
    """Verify directed BFS reasoning path across multiple verified provisions."""
    engine = RegulatoryGraphEngine()
    engine.add_provision("ProvA", {})
    engine.add_provision("ProvB", {})
    engine.add_provision("ProvC", {})
    engine.add_provision("ProvD", {})

    engine.add_edge_with_provenance("e1", "ProvA", "ProvB", "references", 1, 2, "Ref to ProvB")
    engine.add_edge_with_provenance("e2", "ProvB", "ProvC", "modifies", 2, 3, "Modifies ProvC")
    engine.add_edge_with_provenance("e3", "ProvC", "ProvD", "excepts", 3, 4, "Except ProvD")

    path = engine.find_reasoning_path("ProvA", "ProvD")
    assert len(path) == 3
    assert [e.edge_id for e in path] == ["e1", "e2", "e3"]


# ==============================================================================
# Scenario 10: Multi-Passage Cross-Reference Resolution
# ==============================================================================

def test_multi_passage_assembler_resolution(mock_regulatory_index: DocumentIndex):
    """MultiPassageAssembler must resolve referenced sections without hallucinating."""
    assembler = MultiPassageAssembler(mock_regulatory_index)

    # Chunk p1 references Section 4
    evidence = assembler.assemble_multi_passage_evidence("chunk_p1_c0", max_hops=2)
    assert "chunk_p3_c0" in evidence.connected_chunk_ids
    assert any("Section 4" in ref for ref in evidence.cross_references_resolved)

    # Non-existent reference is cleanly marked unresolved
    unresolved_test = assembler.resolve_reference("Section 9999")
    assert unresolved_test == []


# ==============================================================================
# Scenario 11: Table Reasoner Column Association Safety
# ==============================================================================

def test_table_reasoner_column_association(mock_regulatory_index: DocumentIndex):
    """Ensure classification respects column placement (Appendix I vs II), not page co-occurrence."""
    reasoner = TableReasoner(mock_regulatory_index)

    # Panthera leo is in Column 1 (Appendix I)
    leo_cells = reasoner.lookup_taxon_cell("Panthera leo")
    assert len(leo_cells) == 1
    assert leo_cells[0].column_heading == "Appendix I"
    assert leo_cells[0].pdf_page_index == 2

    # Felis catus is in Column 2 (Appendix II)
    catus_cells = reasoner.lookup_taxon_cell("Felis catus")
    assert len(catus_cells) == 1
    assert catus_cells[0].column_heading == "Appendix II"
    assert catus_cells[0].pdf_page_index == 2


# ==============================================================================
# Scenario 12: Footnote Attachment to Table Cell
# ==============================================================================

def test_table_footnote_attachment(mock_regulatory_index: DocumentIndex):
    """Ensure modifying footnotes in table cells are captured in TableCellProvenance."""
    reasoner = TableReasoner(mock_regulatory_index)
    ursus_cells = reasoner.lookup_taxon_cell("Ursus arctos")
    assert len(ursus_cells) == 1
    assert "#1" in ursus_cells[0].modifying_footnotes


# ==============================================================================
# Scenario 13: Live CITES Primates Appendix II vs I Classification
# ==============================================================================

def test_live_cites_primates_classification(cites_index: DocumentIndex):
    """
    Test real CITES Appendix classification:
    - PRIMATES spp. is listed in Appendix II with exclusion for Appendix I.
    - Macaca silenus is explicitly in Appendix I (direct listing).
    - Macaca mulatta is NOT in Appendix I, so it conditionally falls under Appendix II
      when higher_taxon="PRIMATES" is provided.
    """
    reasoner = TableReasoner(cites_index)

    # 1. Macaca silenus explicitly in Appendix I (Direct Listing)
    res_silenus = reasoner.resolve_cites_classification("Macaca silenus")
    assert res_silenus["classification"] == "Appendix I"
    assert res_silenus["status"] == "direct_listing"
    assert res_silenus["is_explicit_listing"] is True
    assert res_silenus["is_inference"] is False
    assert res_silenus["provenance"]["pdf_page_index"] == 20

    # 2. Macaca mulatta conditionally inferred via higher taxon
    res_mulatta = reasoner.resolve_cites_classification("Macaca mulatta", higher_taxon="PRIMATES")
    assert "Appendix II" in res_mulatta["classification"]
    assert res_mulatta["status"] == "conditional_inference"
    assert res_mulatta["is_explicit_listing"] is False
    assert res_mulatta["is_inference"] is True
    assert res_mulatta["provenance"]["pdf_page_index"] == 19
    assert len(res_mulatta["assumptions"]) > 0
    assert "PRIMATES spp." in res_mulatta["rationale"]


def test_regression_literal_text_absence(cites_index: DocumentIndex):
    """
    Verify Question 1: Does the literal text 'Macaca mulatta' or 'Rhesus' occur in the supplied PDF index?
    Must be strictly False.
    """
    all_page_text = " ".join([p.text for p in cites_index.pages]).lower()
    all_table_text = " ".join([cell for p in cites_index.pages for t in p.tables for r in t.rows for cell in r]).lower()

    assert "macaca mulatta" not in all_page_text
    assert "rhesus" not in all_page_text
    assert "macaca mulatta" not in all_table_text
    assert "rhesus" not in all_table_text


def test_regression_strict_document_only_policy(cites_index: DocumentIndex):
    """
    Verify Question 7: In strict document-only mode, the system refuses to assume
    external taxonomy and returns unlisted_in_document (insufficient evidence).
    """
    reasoner = TableReasoner(cites_index)
    res = reasoner.resolve_cites_classification(
        "Macaca mulatta",
        higher_taxon="PRIMATES",
        strict_document_only=True,
    )
    assert res["status"] == "unlisted_in_document"
    assert res["is_explicit_listing"] is False
    assert res["is_inference"] is False
    assert "Strict document-only policy" in res["rationale"]
    assert res["provenance"] is None


def test_regression_unlisted_species_without_higher_taxon(cites_index: DocumentIndex):
    """
    Verify that querying an unlisted species without higher taxon premise
    does NOT default to PRIMATES and returns unlisted_in_document.
    """
    reasoner = TableReasoner(cites_index)
    res_mulatta = reasoner.resolve_cites_classification("Macaca mulatta", higher_taxon=None)
    assert res_mulatta["status"] == "unlisted_in_document"
    assert res_mulatta["classification"] == "Unlisted / Unknown"
    assert res_mulatta["provenance"] is None

    res_unicorn = reasoner.resolve_cites_classification("RandomUnicorn", higher_taxon=None)
    assert res_unicorn["status"] == "unlisted_in_document"
    assert res_unicorn["classification"] == "Unlisted / Unknown"


def test_regression_unsupported_taxonomic_mapping(cites_index: DocumentIndex):
    """
    Verify that an unlisted species with an unsupported/non-existent higher taxon
    returns unsupported_taxonomic_mapping.
    """
    reasoner = TableReasoner(cites_index)
    res = reasoner.resolve_cites_classification("RandomUnicorn", higher_taxon="MythicalBeasts")
    assert res["status"] == "unsupported_taxonomic_mapping"
    assert res["classification"] == "Unlisted / Unknown"
    assert res["is_inference"] is False
    assert any("MythicalBeasts" in a for a in res["assumptions"])



# ==============================================================================
# Scenario 14: Live CITES Pharmaceutical Research Unstated
# ==============================================================================

def test_live_cites_pharmaceutical_research_unstated(cites_index: DocumentIndex, evaluator: DeterministicRuleEvaluator):
    """
    Verify that pharmaceutical research permits or Indian central government approvals
    are evaluated as UNKNOWN due to missing evidence.
    """
    # Check that search for pharmaceutical yields no results
    pharm_chunks = [c for c in cites_index.chunks if "pharmaceutical" in c.text.lower()]
    assert len(pharm_chunks) == 0

    # Rule requiring Indian ministry approval for pharmaceutical export
    rule = RegulatoryRule(
        rule_id="R_CITES_EXPORT_PERMIT",
        rule_type="requirement",
        subject="Rhesus Monkey Export",
        predicate="authorized for pharmaceutical research in USA",
        operator=LogicalOperator.AND,
        conditions=[
            RuleCondition(
                condition_id="C_APP_II",
                description="Species governed by Appendix II provisions",
                state=ConditionState.SATISFIED,
                source_page=19,
            ),
            RuleCondition(
                condition_id="C_PHARM_PERMIT",
                description="Special pharmaceutical research export permit authorized",
                state=ConditionState.UNKNOWN,  # Not found in document
            ),
            RuleCondition(
                condition_id="C_INDIAN_MINISTRY",
                description="Indian central-government ministry approvals identified",
                state=ConditionState.UNKNOWN,  # Not found in document
            ),
        ],
    )
    res = evaluator.evaluate_rule(rule)
    assert res.status == "unknown"
    assert len(res.missing_evidence) == 2


# ==============================================================================
# Scenario 15: Footnote #4 on Euphorbia spp. in Live CITES Index
# ==============================================================================

def test_live_cites_euphorbia_footnote(cites_index: DocumentIndex):
    """Verify that footnote #4 is detected on Euphorbia spp. in CITES tables."""
    reasoner = TableReasoner(cites_index)
    euphorbia_cells = reasoner.lookup_taxon_cell("Euphorbia spp.")
    assert len(euphorbia_cells) > 0
    assert any("#4" in cell.modifying_footnotes for cell in euphorbia_cells)


# ==============================================================================
# Scenario 16: Serialization Round-Trip
# ==============================================================================

def test_serialization_roundtrip():
    """Ensure RegulatoryRule, RuleCondition, and RuleException serialize and deserialize losslessly."""
    cond = RuleCondition(
        condition_id="C1",
        description="Has import permit",
        is_mandatory=True,
        state=ConditionState.SATISFIED,
        citation_quote="Import permit required",
        source_page=5,
    )
    exc = RuleException(
        exception_id="EX1",
        description="Personal baggage exemption",
        applies_when="Under 2kg",
        state=ConditionState.NOT_SATISFIED,
    )
    rule = RegulatoryRule(
        rule_id="R1",
        rule_type="requirement",
        subject="Importer",
        predicate="cleared",
        operator=LogicalOperator.AND,
        conditions=[cond],
        exceptions=[exc],
        min_threshold_count=1,
    )

    d = rule.to_dict()
    assert d["rule_id"] == "R1"
    assert d["conditions"][0]["condition_id"] == "C1"
    assert d["exceptions"][0]["exception_id"] == "EX1"

    # Deserialization test
    cond_restored = RuleCondition.from_dict(d["conditions"][0])
    assert cond_restored.condition_id == cond.condition_id
    assert cond_restored.state == ConditionState.SATISFIED
    assert cond_restored.is_mandatory is True

    exc_restored = RuleException.from_dict(d["exceptions"][0])
    assert exc_restored.exception_id == exc.exception_id
    assert exc_restored.state == ConditionState.NOT_SATISFIED


# ==============================================================================
# Scenario 17: QA Engine Integration with Multi-Passage Reasoning
# ==============================================================================

def test_qa_engine_with_phase4_integration(mock_regulatory_index: DocumentIndex):
    """Verify DocumentQAEngine functions cleanly with multi-passage evidence."""
    engine = DocumentQAEngine(mock=True)
    response = engine.answer_question(
        index=mock_regulatory_index,
        question="What permits are required under Section 1?",
    )
    assert response.status == QAStatus.ANSWERED.value
    assert len(response.evidence) > 0
    assert response.evidence[0].pdf_page_index in [1, 2, 3]
