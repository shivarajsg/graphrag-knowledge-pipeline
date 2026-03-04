# GraphRAG Document Intelligence & Knowledge Graph Pipeline

A production-grade, document-grounded AI system combining **Knowledge Graph construction (GraphRAG)**, **PDF ingestion with layout preservation**, **evidence-grounded question answering**, **claim-level citation integrity & NLI verification**, **regulatory table reasoning**, and an **interactive Streamlit web application**.

---

## Key Capabilities (Phase 0 – 5)

1. **Document Ingestion & Indexing (Phase 1)**: Native PDF text and table extraction (`PyMuPDF`), physical page tracking, character offsets, layout-block detection, and content-hash caching.
2. **Evidence-Grounded Question Answering (Phase 2)**: Strict document-only question answering with controlled response statuses (`answered`, `partially_answered`, `insufficient_evidence`, `conflicting_evidence`).
3. **Claim-Level Evidence Verification (Phase 3)**: Independent citation validation, verbatim quotation matching, and semantic NLI contradiction safeguards.
4. **Advanced Regulatory & Table Reasoning (Phase 4)**: Three-valued logic (`SATISFIED`, `NOT_SATISFIED`, `UNKNOWN`), table column-association preservation (e.g. CITES Appendix I vs II), modifying footnotes, and explicit labeling of conditional taxonomic inferences.
5. **Document Intelligence Web Application (Phase 5)**: Local Streamlit web application (`app.py`) for non-programmer users, featuring verbatim source inspection, PDF page rendering, and regulatory table inspection.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │          Uploaded PDF Document         │
                      └───────────────────┬────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  PDFLoader (PyMuPDF)│
                               │  Text & Table Parser│
                               └──────────┬──────────┘
                                          │
                      ┌───────────────────▼────────────────────┐
                      │    DocumentChunker & DocumentIndex     │
                      │  (Deterministic Chunks & Provenance)   │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
          ┌─────────▼─────────┐ ┌─────────▼─────────┐ ┌─────────▼─────────┐
          │   TableReasoner   │ │MultiPassageAssembl│ │  GraphQueryEngine │
          │ (CITES Appendices)│ │ (Cross-References)│ │  (Knowledge Graph)│
          └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  DocumentQAEngine   │
                               │(Passage Retrieval & │
                               │ Grounded Generation)│
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  CitationValidator  │
                               │ (Verbatim Provenance│
                               │   & Page Bounds)    │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │ClaimVerifierOrchestr│
                               │ (NLI Verification & │
                               │Contradiction Checks)│
                               └──────────┬──────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
          ┌─────────▼─────────┐                       ┌─────────▼─────────┐
          │   Streamlit Web   │                       │    Unified CLI    │
          │   App (app.py)    │                       │  (cli.py / main)  │
          └───────────────────┘                       └───────────────────┘
```

---

## Installation & Environment Setup

### 1. Requirements
- Python 3.10+ (tested on Python 3.11)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### 2. Install Dependencies
```bash
git clone https://github.com/shivarajsg/graphrag-knowledge-pipeline.git
cd graphrag-knowledge-pipeline
pip install -r requirements.txt
```

### 3. API Key Configuration (Optional for Live Mode)
The application includes a fully functional, offline **Deterministic Mock Engine** for testing with zero API costs.

To enable **Live Anthropic Claude** generation:
```bash
# On Linux/macOS:
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_MODEL="claude-sonnet-5-5"

# On Windows (PowerShell):
$env:ANTHROPIC_API_KEY="your-anthropic-api-key"
$env:ANTHROPIC_MODEL="claude-sonnet-5-5"
```

---

## Launching the Web Application (Phase 5)

Start the interactive Streamlit interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Key Web UI Features:
1. **Document Ingestion**:
   - Upload any digital PDF (up to 50 MB).
   - Fast content-hash caching avoids reprocessing unchanged documents.
   - Or click **Load CITES Appendices (Instant Load)** to load the bundled 76-page treaty index.
2. **Strict Document-Only Mode**:
   - Guarantees answers derive strictly from the active document.
   - Prevents outside hallucinations or unverified external knowledge.
3. **Claim Verification Badges**:
   - Each claim is individually labeled: `VERIFIED SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `INCONCLUSIVE`, or `NOT CHECKED`.
4. **Verbatim Evidence & PDF Page Viewer**:
   - View exact source quotes and page numbers.
   - Click **View Physical Page in PDF** to render and inspect the original PDF page image inline.
5. **Regulatory Table Reasoner (Phase 4)**:
   - Search species (e.g. *Macaca silenus*, *Macaca mulatta*, *Euphorbia spp.*).
   - Preserves table column associations (Appendix I vs II).
   - Distinguishes direct document facts from conditional taxonomic inferences.

---

## Command-Line Interface (CLI)

The repository provides a rich set of CLI tools for document processing, search, question answering, and table reasoning.

### 1. PDF Ingestion & Indexing
```bash
# Ingest PDF and build index
python -m graphrag.ingestion.cli --pdf CITES_E-Appendices.pdf --output outputs/cites_index.json
```

### 2. Search Indexed Document
```bash
# Search for keywords
python -m graphrag.ingestion.cli --index outputs/cites_index.json --query "PRIMATES monkeys" --top-k 5
```

### 3. Ask Questions via CLI
```bash
# Ask question using deterministic mock engine
python -m graphrag.ingestion.cli --index outputs/cites_index.json --ask "Can Rhesus Monkeys be exported to the USA for pharmaceutical research?" --mock

# Ask question using live Anthropic Claude
python -m graphrag.ingestion.cli --index outputs/cites_index.json --ask "What are the rules for trade in Felidae species?" --api-key "$ANTHROPIC_API_KEY"
```

### 4. Regulatory Table & Taxon Reasoning
```bash
# Direct table listing (e.g. Macaca silenus in Appendix I)
python -m graphrag.ingestion.cli --index outputs/cites_index.json --table-lookup "Macaca silenus"

# Unlisted taxon without external premise (returns insufficient evidence)
python -m graphrag.ingestion.cli --index outputs/cites_index.json --table-lookup "Macaca mulatta"

# Conditional taxonomic inference with explicit higher-taxon premise
python -m graphrag.ingestion.cli --index outputs/cites_index.json --table-lookup "Macaca mulatta" --higher-taxon "PRIMATES"

# Strict document-only policy (rejects unverified taxonomic assumptions)
python -m graphrag.ingestion.cli --index outputs/cites_index.json --table-lookup "Macaca mulatta" --higher-taxon "PRIMATES" --strict
```

### 5. Multi-Passage Cross-Reference Tracing
```bash
# Trace cross-references, annotations, and footnotes starting from a chunk ID
python -m graphrag.ingestion.cli --index outputs/cites_index.json --cross-refs doc_0c488ede5f9b_p065_c01
```

### 6. Original Knowledge Graph Pipeline
```bash
# Run entity extraction and graph building
python main.py

# Query the knowledge graph
python main.py --query "products launched in 2020"
```

---

## Automated Test Suite

The project includes **152 comprehensive automated tests** across all phases:

| Phase | Test File | Test Count | Description |
| :--- | :--- | :--- | :--- |
| **Phase 0** | `tests/test_pipeline.py` | 60 | Baseline GraphRAG pipeline, extraction, and confidence scoring |
| **Phase 1** | `tests/test_ingestion.py` | 9 | PDF loading, page extraction, chunking, and index serialization |
| **Phase 2** | `tests/test_qa_engine.py` | 18 | Evidence-grounded QA, citation validation, and abstention |
| **Phase 3** | `tests/test_qa_verifier.py` | 21 | Claim-level NLI verification and hallucination safeguards |
| **Phase 4** | `tests/test_rule_engine.py` | 24 | Three-valued logic, table reasoner, provision graph, and CITES tests |
| **Phase 5** | `tests/test_app_service.py` | 20 | Service layer, caching, isolation, and PDF page rendering |
| **Total** | | **152** | **100% Pass Rate** |

Run the full test suite:
```bash
python -m pytest tests/
```

---

## Known Limitations & Security Notes

1. **Scanned PDF / OCR Limitations**:
   - The document ingestion pipeline utilizes native PDF text and font extraction via PyMuPDF.
   - Scanned image-only PDFs without an embedded OCR text layer are flagged as `is_suspicious_scanned` and generate extraction warnings. OCR execution requires external Tesseract / OCR engines.
2. **Strict Document-Only Grounding**:
   - When information is absent from the supplied document (e.g., export permits for pharmaceutical research or specific Indian central-government approvals), the system is designed to **abstain** with `insufficient_evidence` rather than invent facts.
3. **Taxonomic Inferences vs. Direct Facts**:
   - Unlisted species are never assumed to belong to a higher taxon unless explicitly supplied as an external premise. When an external premise is provided in non-strict mode, it is clearly flagged as a `CONDITIONAL TAXONOMIC INFERENCE (NOT IN DOCUMENT VERBATIM)`.
4. **Data Isolation**:
   - Uploaded documents and generated indexes are stored locally under `outputs/uploads/` and `outputs/indexes/`.
   - Switching active documents in the UI resets the session state and invalidates prior QA responses, preventing cross-document contamination.

---

## Author & Maintainer

Shivraj  
Repository: [graphrag-knowledge-pipeline](https://github.com/shivarajsg/graphrag-knowledge-pipeline)
