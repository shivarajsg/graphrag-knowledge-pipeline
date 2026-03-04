"""
graphrag – A reusable Graph RAG pipeline mini-library.

Public API
----------
    from graphrag.extractor    import extract
    from graphrag.graph_builder import build_graph
    from graphrag.renderer     import render_mermaid
    from graphrag.retriever    import retrieve_chunks
    from graphrag.pipeline     import run_pipeline
"""

__version__ = "4.0.0"

from .extractor      import extract
from .graph_builder  import build_graph
from .renderer       import render_mermaid
from .retriever      import retrieve_chunks
from .pipeline       import run_pipeline
from .classifier     import classify_entity_type, classify_entities
from .query_engine   import GraphQueryEngine, load_engine
from .ingestion      import PDFLoader, DocumentChunker, DocumentIndex, index_pdf, search_document_index
from .qa_schemas     import QAStatus, ClaimVerificationStatus, EvidenceItem, Claim, QAResponse
from .qa_validator   import CitationValidator
from .qa_verifier    import (
    ClaimVerificationResult,
    BaseSemanticVerifier,
    RuleBasedSemanticVerifier,
    LLMClaimVerifier,
    InactiveSemanticVerifier,
    ClaimVerifierOrchestrator,
)
from .qa_engine      import DocumentQAEngine, BaseQAClient, AnthropicQAClient, MockQAClient
from .regulatory_schemas import (
    ConditionState,
    LogicalOperator,
    RuleCondition,
    RuleException,
    RegulatoryRule,
    ProvisionEdge,
    ProvisionGraph,
    TableCellProvenance,
    RuleEvaluationResult,
    MultiPassageEvidence,
    TaxonResolutionResult,
)

from .rule_engine    import (
    DeterministicRuleEvaluator,
    MultiPassageAssembler,
    TableReasoner,
    RegulatoryGraphEngine,
)
from .app_service    import DocumentIntelligenceService


__all__ = [
    "extract",
    "build_graph",
    "render_mermaid",
    "retrieve_chunks",
    "run_pipeline",
    "classify_entity_type",
    "classify_entities",
    "GraphQueryEngine",
    "load_engine",
    "PDFLoader",
    "DocumentChunker",
    "DocumentIndex",
    "index_pdf",
    "search_document_index",
    "QAStatus",
    "ClaimVerificationStatus",
    "EvidenceItem",
    "Claim",
    "QAResponse",
    "CitationValidator",
    "ClaimVerificationResult",
    "BaseSemanticVerifier",
    "RuleBasedSemanticVerifier",
    "LLMClaimVerifier",
    "InactiveSemanticVerifier",
    "ClaimVerifierOrchestrator",
    "DocumentQAEngine",
    "BaseQAClient",
    "AnthropicQAClient",
    "MockQAClient",
    "ConditionState",
    "LogicalOperator",
    "RuleCondition",
    "RuleException",
    "RegulatoryRule",
    "ProvisionEdge",
    "ProvisionGraph",
    "TableCellProvenance",
    "RuleEvaluationResult",
    "MultiPassageEvidence",
    "TaxonResolutionResult",
    "DeterministicRuleEvaluator",
    "MultiPassageAssembler",
    "TableReasoner",
    "RegulatoryGraphEngine",
    "DocumentIntelligenceService",
]






