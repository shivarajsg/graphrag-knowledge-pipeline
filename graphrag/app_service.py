"""
graphrag.app_service
--------------------
Phase 5: Application Service Layer for Document Intelligence & Evidence-Grounded QA.
Decouples business logic from Streamlit presentation to ensure full testability.
Handles:
1. Safe PDF upload validation, caching (content-hash based), and indexing.
2. Active document index management with strict session isolation.
3. Evidence-grounded Question Answering and claim verification invocation.
4. Physical PDF page rendering using PyMuPDF (fitz) for verbatim visual verification.
5. Regulatory table reasoning and taxon lookup integration.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from .ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .ingestion.indexer import index_pdf
from .ingestion.schemas import DocumentIndex, ExtractedPage
from .qa_engine import DEFAULT_QA_MODEL, DocumentQAEngine
from .qa_schemas import ClaimVerificationStatus, QAResponse, QAStatus
from .rule_engine import TableReasoner

logger = logging.getLogger(__name__)

# Configurable limits
MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB limit
ALLOWED_EXTENSIONS: set = {".pdf"}


class DocumentIntelligenceService:
    """
    Service layer orchestrating document ingestion, caching, active index management,
    evidence-grounded question answering, and PDF page rendering.
    """

    def __init__(self, base_storage_dir: str | Path = "outputs") -> None:
        self.base_dir = Path(base_storage_dir)
        self.uploads_dir = self.base_dir / "uploads"
        self.indexes_dir = self.base_dir / "indexes"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self.active_index: Optional[DocumentIndex] = None
        self.active_pdf_path: Optional[Path] = None
        self.active_doc_hash: Optional[str] = None
        self.last_response: Optional[QAResponse] = None
        self.last_query: Optional[str] = None

    # ── Document Upload & Ingestion ───────────────────────────────────────────

    def validate_upload(self, file_bytes: bytes, filename: str) -> None:
        """
        Validates uploaded file format, size, and PDF integrity.
        Raises ValueError with user-friendly message on any validation error.
        """
        if not filename:
            raise ValueError("Filename is required.")

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Please upload a standard PDF document (.pdf).")

        size = len(file_bytes)
        if size == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")
        if size > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            raise ValueError(f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit of {max_mb} MB.")

        # Test PDF header and parsing integrity via PyMuPDF
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.is_encrypted:
                doc.close()
                raise ValueError("PDF is password-protected or encrypted. Please remove encryption before uploading.")
            if len(doc) == 0:
                doc.close()
                raise ValueError("PDF document contains 0 pages.")
            doc.close()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted PDF file: {exc}")

    def upload_and_index_document(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> Tuple[DocumentIndex, bool]:
        """
        Ingests and indexes an uploaded PDF document:
        1. Validates file integrity and size.
        2. Computes deterministic SHA-256 content hash.
        3. If an index for this content hash already exists, safely reuses it without re-indexing.
        4. Otherwise, writes file to uploads storage and creates a searchable DocumentIndex.
        5. Updates active document state and resets any stale QA responses.

        Returns (DocumentIndex, is_cached).
        """
        self.validate_upload(file_bytes=file_bytes, filename=filename)

        doc_hash = hashlib.sha256(file_bytes).hexdigest()
        clean_name = Path(filename).name
        saved_pdf_path = self.uploads_dir / f"{doc_hash[:12]}_{clean_name}"
        saved_index_path = self.indexes_dir / f"{doc_hash}.json"

        is_cached = False

        # If unchanged index exists on disk, load directly
        if saved_index_path.exists() and saved_pdf_path.exists():
            logger.info(f"[app_service] Content hash match ({doc_hash[:12]}). Reusing existing index: {saved_index_path}")
            index = DocumentIndex.load(saved_index_path)
            is_cached = True
        else:
            # Write PDF to storage
            saved_pdf_path.write_bytes(file_bytes)
            logger.info(f"[app_service] Saved upload to {saved_pdf_path}. Indexing PDF...")

            # Run Phase 1 PDF indexing
            index = index_pdf(
                pdf_path=saved_pdf_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                output_path=saved_index_path,
            )

        # Set as active document and isolate session state
        self._set_active_document(index=index, pdf_path=saved_pdf_path, doc_hash=doc_hash)
        return index, is_cached

    def load_existing_index(
        self,
        index_path: str | Path,
        pdf_path: Optional[str | Path] = None,
    ) -> DocumentIndex:
        """
        Loads a pre-computed DocumentIndex from disk (e.g. CITES Appendices index).
        Sets it as the active document and isolates session state.
        """
        idx_p = Path(index_path)
        if not idx_p.exists():
            raise FileNotFoundError(f"Document index file not found: {idx_p}")

        index = DocumentIndex.load(idx_p)

        actual_pdf_path = Path(pdf_path) if pdf_path else None
        if not actual_pdf_path or not actual_pdf_path.exists():
            # Check relative to index or project root
            candidate = Path(index.source_file)
            if candidate.exists():
                actual_pdf_path = candidate
            elif (self.base_dir.parent / index.source_file).exists():
                actual_pdf_path = self.base_dir.parent / index.source_file

        self._set_active_document(index=index, pdf_path=actual_pdf_path, doc_hash=index.document_hash)
        return index

    def clear_active_document(self) -> None:
        """Resets the active document, clearing index, citations, and answers."""
        self.active_index = None
        self.active_pdf_path = None
        self.active_doc_hash = None
        self.last_response = None
        self.last_query = None
        logger.info("[app_service] Active document session cleared.")

    def _set_active_document(
        self,
        index: DocumentIndex,
        pdf_path: Optional[Path],
        doc_hash: str,
    ) -> None:
        """Internal setter isolating session state when active document changes."""
        self.active_index = index
        self.active_pdf_path = pdf_path
        self.active_doc_hash = doc_hash
        # Invalidate previous QA response to avoid cross-document contamination
        self.last_response = None
        self.last_query = None

    # ── Document Inspection & Summary ─────────────────────────────────────────

    def get_document_summary(self) -> Dict[str, Any]:
        """Returns structured metadata summary of active document."""
        if not self.active_index:
            return {"has_active_document": False}

        idx = self.active_index
        total_tables = sum(len(p.tables) for p in idx.pages)
        scanned_pages = [p.pdf_page_index for p in idx.pages if p.is_suspicious_scanned]
        pages_with_warnings = [p.pdf_page_index for p in idx.pages if p.warnings]

        file_size_bytes = 0
        if self.active_pdf_path and self.active_pdf_path.exists():
            file_size_bytes = self.active_pdf_path.stat().st_size

        return {
            "has_active_document": True,
            "document_id": idx.document_id,
            "source_file": idx.source_file,
            "document_hash": idx.document_hash,
            "file_size_bytes": file_size_bytes,
            "total_pages": idx.total_pages,
            "total_chunks": idx.total_chunks,
            "total_tables": total_tables,
            "scanned_pages_count": len(scanned_pages),
            "scanned_pages": scanned_pages,
            "pages_with_warnings_count": len(pages_with_warnings),
            "pages_with_warnings": pages_with_warnings,
            "index_version": idx.index_version,
            "created_at": idx.created_at,
        }

    def get_total_pages(self) -> int:
        """Returns the total physical pages in the active document."""
        if self.active_index and self.active_index.total_pages > 0:
            return self.active_index.total_pages
        if self.active_pdf_path and self.active_pdf_path.exists():
            try:
                doc = fitz.open(self.active_pdf_path)
                count = len(doc)
                doc.close()
                return count
            except Exception:
                pass
        return 0

    # ── Question Answering Execution ─────────────────────────────────────────

    def submit_question(
        self,
        question: str,
        mock: bool = True,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_k: int = 5,
    ) -> QAResponse:
        """
        Executes evidence-grounded question answering on the active document index.
        Runs full retrieval, prompt generation, citation validation, and semantic verification.
        """
        if not self.active_index:
            raise ValueError("No active document loaded. Please upload and index a PDF first.")

        clean_q = question.strip() if question else ""
        if not clean_q:
            raise ValueError("Question string cannot be empty.")

        effective_model = model or DEFAULT_QA_MODEL

        # Instantiate DocumentQAEngine with requested execution mode
        qa_engine = DocumentQAEngine(
            mock=mock,
            api_key=api_key,
            model=effective_model,
            default_top_k=top_k,
        )

        response = qa_engine.answer_question(
            index=self.active_index,
            question=clean_q,
            top_k=top_k,
        )

        self.last_response = response
        self.last_query = clean_q
        return response

    # ── Source PDF Page Rendering ─────────────────────────────────────────────

    def render_page_image(self, pdf_page_index: int, zoom: float = 1.5) -> Optional[bytes]:
        """
        Renders a physical PDF page as a PNG image for verbatim visual inspection.
        Returns PNG bytes, or None if PDF file is not available on disk.
        """
        if not self.active_pdf_path or not self.active_pdf_path.exists():
            logger.warning("[app_service] Cannot render page image: active PDF file is not found.")
            return None

        try:
            doc = fitz.open(self.active_pdf_path)
            # Physical page index is 1-based; fitz is 0-based
            page_idx = pdf_page_index - 1
            if page_idx < 0 or page_idx >= len(doc):
                doc.close()
                logger.warning(f"[app_service] Page index {pdf_page_index} out of range (1..{len(doc)}).")
                return None

            page = doc[page_idx]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            doc.close()
            return png_bytes
        except Exception as exc:
            logger.error(f"[app_service] Error rendering PDF page {pdf_page_index}: {exc}")
            return None

    # ── Regulatory Table & Taxonomy Reasoning (Phase 4) ───────────────────────

    def inspect_table_taxon(
        self,
        taxon_name: str,
        higher_taxon: Optional[str] = None,
        strict_document_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes Phase 4 TableReasoner on active document.
        Preserves column associations and distinguishes direct listings from conditional inferences.
        """
        if not self.active_index:
            raise ValueError("No active document loaded.")

        clean_name = taxon_name.strip() if taxon_name else ""
        if not clean_name:
            raise ValueError("Taxon name cannot be empty.")

        reasoner = TableReasoner(self.active_index)
        return reasoner.resolve_cites_classification(
            species_name=clean_name,
            higher_taxon=higher_taxon,
            strict_document_only=strict_document_only,
        )
