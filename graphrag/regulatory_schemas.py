"""
graphrag.regulatory_schemas
---------------------------
Phase 4: Advanced Document Reasoning, Regulatory Rules, Exceptions, and Graph Evidence.
Data models for regulatory provisions, structured conditions, three-valued logic,
exceptions, cross-references, tables, and provision provenance graphs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ConditionState(str, Enum):
    """
    Three-valued logic states for regulatory conditions.
    Never assume an unknown condition is not_satisfied without evidence.
    """
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNKNOWN = "unknown"


class LogicalOperator(str, Enum):
    """Logical operators for combining conditions in regulatory rules."""
    AND = "AND"
    OR = "OR"
    THRESHOLD = "THRESHOLD"
    NONE = "NONE"


@dataclass
class RuleCondition:
    """
    An individual condition or subcondition within a regulatory rule.
    """
    condition_id: str
    description: str
    is_mandatory: bool = False
    state: ConditionState = ConditionState.UNKNOWN
    subconditions: List[RuleCondition] = field(default_factory=list)
    operator: LogicalOperator = LogicalOperator.NONE
    evidence_chunk_ids: List[str] = field(default_factory=list)
    citation_quote: Optional[str] = None
    source_page: Optional[int] = None
    group_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "description": self.description,
            "is_mandatory": self.is_mandatory,
            "state": self.state.value if isinstance(self.state, ConditionState) else str(self.state),
            "subconditions": [sc.to_dict() for sc in self.subconditions],
            "operator": self.operator.value if isinstance(self.operator, LogicalOperator) else str(self.operator),
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "citation_quote": self.citation_quote,
            "source_page": self.source_page,
            "group_name": self.group_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuleCondition:
        state_val = data.get("state", ConditionState.UNKNOWN.value)
        try:
            state = ConditionState(state_val)
        except ValueError:
            state = ConditionState.UNKNOWN

        op_val = data.get("operator", LogicalOperator.NONE.value)
        try:
            op = LogicalOperator(op_val)
        except ValueError:
            op = LogicalOperator.NONE

        return cls(
            condition_id=str(data.get("condition_id", "")),
            description=str(data.get("description", "")),
            is_mandatory=bool(data.get("is_mandatory", False)),
            state=state,
            subconditions=[cls.from_dict(sc) for sc in data.get("subconditions", [])],
            operator=op,
            evidence_chunk_ids=list(data.get("evidence_chunk_ids", [])),
            citation_quote=data.get("citation_quote"),
            source_page=data.get("source_page"),
            group_name=data.get("group_name"),
        )


@dataclass
class RuleException:
    """
    An explicit exception, exemption, or exclusion that overrides or modifies a rule.
    """
    exception_id: str
    description: str
    applies_when: str
    state: ConditionState = ConditionState.UNKNOWN  # Does this exception apply?
    evidence_chunk_ids: List[str] = field(default_factory=list)
    citation_quote: Optional[str] = None
    source_page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_id": self.exception_id,
            "description": self.description,
            "applies_when": self.applies_when,
            "state": self.state.value if isinstance(self.state, ConditionState) else str(self.state),
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "citation_quote": self.citation_quote,
            "source_page": self.source_page,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuleException:
        state_val = data.get("state", ConditionState.UNKNOWN.value)
        try:
            state = ConditionState(state_val)
        except ValueError:
            state = ConditionState.UNKNOWN

        return cls(
            exception_id=str(data.get("exception_id", "")),
            description=str(data.get("description", "")),
            applies_when=str(data.get("applies_when", "")),
            state=state,
            evidence_chunk_ids=list(data.get("evidence_chunk_ids", [])),
            citation_quote=data.get("citation_quote"),
            source_page=data.get("source_page"),
        )


@dataclass
class RegulatoryRule:
    """
    Structured representation of a document rule, statutory standard, or policy clause.
    Retains full source provenance and tracks evaluation outcomes.
    """
    rule_id: str
    rule_type: str                            # permission | prohibition | requirement | classification | exemption
    subject: str
    predicate: str
    operator: LogicalOperator = LogicalOperator.AND
    conditions: List[RuleCondition] = field(default_factory=list)
    exceptions: List[RuleException] = field(default_factory=list)
    min_threshold_count: Optional[int] = None # e.g. at least 4 qualifying conditions
    mandatory_group_counts: Dict[str, int] = field(default_factory=dict) # group_name -> required min
    source_document: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    source_passages: List[str] = field(default_factory=list)
    physical_pages: List[int] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)
    is_inferred: bool = False
    evaluation_result: Optional[str] = None   # satisfied | not_satisfied | unknown | exempted
    evaluation_trail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "operator": self.operator.value if isinstance(self.operator, LogicalOperator) else str(self.operator),
            "conditions": [c.to_dict() for c in self.conditions],
            "exceptions": [e.to_dict() for e in self.exceptions],
            "min_threshold_count": self.min_threshold_count,
            "mandatory_group_counts": dict(self.mandatory_group_counts),
            "source_document": self.source_document,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_passages": list(self.source_passages),
            "physical_pages": list(self.physical_pages),
            "cross_references": list(self.cross_references),
            "unresolved_references": list(self.unresolved_references),
            "is_inferred": self.is_inferred,
            "evaluation_result": self.evaluation_result,
            "evaluation_trail": list(self.evaluation_trail),
        }


@dataclass
class ProvisionEdge:
    """
    A verified relationship between two provisions or clauses in the document.
    Must have traceable source provenance linking both endpoints.
    """
    edge_id: str
    source_provision: str                     # e.g. chunk_id or section name
    target_provision: str                     # target chunk_id or referenced section name
    relation: str                             # references | excepts | modifies | defines_condition_for | footnote_for
    source_page: int
    target_page: Optional[int] = None
    citation_quote: str = ""
    has_valid_provenance: bool = True
    is_resolved: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvisionGraph:
    """
    Provable graph of document provisions, cross-references, annotations, and exceptions.
    Ensures that edges without valid source provenance cannot establish claims.
    """
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict) # id -> metadata
    edges: List[ProvisionEdge] = field(default_factory=list)

    def add_provision(self, provision_id: str, metadata: Dict[str, Any]) -> None:
        self.nodes[provision_id] = metadata

    def add_edge(self, edge: ProvisionEdge) -> None:
        if not edge.has_valid_provenance:
            logger.warning(f"[provision_graph] Edge '{edge.edge_id}' rejected: missing valid provenance.")
            return
        self.edges.append(edge)

    def get_outbound_edges(self, provision_id: str) -> List[ProvisionEdge]:
        return [e for e in self.edges if e.source_provision == provision_id]

    def get_inbound_edges(self, provision_id: str) -> List[ProvisionEdge]:
        return [e for e in self.edges if e.target_provision == provision_id]

    def resolve_references_for(self, provision_id: str) -> List[ProvisionEdge]:
        return [e for e in self.edges if e.source_provision == provision_id and e.is_resolved]

    def get_unresolved_references(self) -> List[ProvisionEdge]:
        return [e for e in self.edges if not e.is_resolved]


@dataclass
class TableCellProvenance:
    """
    Cell-level provenance preserving row, column association, and modifying footnotes.
    """
    table_id: str
    pdf_page_index: int
    column_heading: str
    row_index: int
    cell_text: str
    modifying_footnotes: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleEvaluationResult:
    """
    Comprehensive result of evaluating a regulatory rule against evidence.
    """
    rule_id: str
    status: str                               # satisfied | not_satisfied | unknown | exempted
    conditions_summary: Dict[str, str] = field(default_factory=dict)
    missing_evidence: List[str] = field(default_factory=list)
    exceptions_applied: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    provenance_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiPassageEvidence:
    """
    Evidence gathered across multiple passages, sections, tables, or footnotes.
    """
    root_chunk_id: str
    connected_chunk_ids: List[str] = field(default_factory=list)
    cross_references_resolved: List[str] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)
    combined_context: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaxonResolutionResult:
    """
    Structured outcome of resolving a taxon in regulatory tables.
    Distinguishes direct document listings from conditional taxonomic inferences.
    """
    species: str
    classification: str                       # e.g. "Appendix I", "Appendix II (Conditional Inference)", "Unlisted / Insufficient Evidence"
    status: str                               # "direct_listing" | "conditional_inference" | "unlisted_in_document" | "unsupported_taxonomic_mapping"
    is_explicit_listing: bool
    is_inference: bool
    rationale: str
    provenance: Optional[TableCellProvenance] = None
    higher_taxon: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    verbatim_source_entry: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "classification": self.classification,
            "status": self.status,
            "is_explicit_listing": self.is_explicit_listing,
            "is_inference": self.is_inference,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "higher_taxon": self.higher_taxon,
            "assumptions": list(self.assumptions),
            "verbatim_source_entry": self.verbatim_source_entry,
        }


