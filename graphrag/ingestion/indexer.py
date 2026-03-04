"""
graphrag.ingestion.indexer
--------------------------
Document indexing and retrieval coordinator.
Connects PDFLoader, DocumentChunker, and the existing multi-backend retriever
into a unified document search pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphrag.retriever import retrieve_chunks

from .chunker import DocumentChunker, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from .pdf_loader import PDFLoader
from .schemas import DocumentChunk, DocumentIndex

logger = logging.getLogger(__name__)


@dataclass
class DocumentSearchResult:
    """
    Search result item with verbatim passage text and complete source citation metadata.
    """
    chunk_id: str
    document_id: str
    source_file: str
    pdf_page_index: int
    printed_page_number: Optional[str]
    section_heading: Optional[str]
    text: str
    page_char_start: int
    page_char_end: int
    score: float
    warnings: List[str]
    extraction_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "pdf_page_index": self.pdf_page_index,
            "printed_page_number": self.printed_page_number,
            "section_heading": self.section_heading,
            "text": self.text,
            "page_char_start": self.page_char_start,
            "page_char_end": self.page_char_end,
            "score": round(self.score, 4),
            "warnings": self.warnings,
            "extraction_method": self.extraction_method,
        }


def index_pdf(
    pdf_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    output_path: Optional[str | Path] = None,
) -> DocumentIndex:
    """
    Ingest a PDF file, split into page-aware chunks, and assemble a DocumentIndex.

    Parameters
    ----------
    pdf_path      : path to source PDF
    chunk_size    : maximum characters per chunk
    chunk_overlap : character overlap between consecutive chunks on the same page
    output_path   : optional file path to save the index JSON

    Returns
    -------
    DocumentIndex
    """
    loader = PDFLoader()
    doc_data = loader.load(pdf_path)

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(
        document_id=doc_data["document_id"],
        source_file=doc_data["source_file"],
        document_hash=doc_data["document_hash"],
        pages=doc_data["pages"],
    )

    index = DocumentIndex(
        document_id=doc_data["document_id"],
        source_file=doc_data["source_file"],
        document_hash=doc_data["document_hash"],
        total_pages=doc_data["total_pages"],
        total_chunks=len(chunks),
        pages=doc_data["pages"],
        chunks=chunks,
        warnings=doc_data["warnings"],
        config={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    if output_path:
        index.save(output_path)
        logger.info(f"[index_pdf] Index saved to {output_path}")

    return index


def search_document_index(
    index: DocumentIndex,
    query: str,
    top_k: int = 5,
) -> List[DocumentSearchResult]:
    """
    Search document chunks in an index using the existing multi-backend retriever.

    Parameters
    ----------
    index  : loaded DocumentIndex
    query  : search query string
    top_k  : number of top passages to return

    Returns
    -------
    List of DocumentSearchResult objects with exact metadata and relevance score.
    """
    if not index.chunks or not query or not query.strip():
        return []

    # Format chunks for graphrag.retriever.retrieve_chunks
    retriever_chunks = index.get_retriever_chunks()

    # Reuse existing multi-stage retriever (sentence-transformers -> sklearn -> pure python)
    raw_results = retrieve_chunks(query=query, chunks=retriever_chunks, top_k=top_k)

    results: List[DocumentSearchResult] = []
    for chunk_dict, score in raw_results:
        results.append(
            DocumentSearchResult(
                chunk_id=chunk_dict.get("chunk_id", chunk_dict.get("id", "")),
                document_id=chunk_dict.get("document_id", index.document_id),
                source_file=chunk_dict.get("source_file", index.source_file),
                pdf_page_index=chunk_dict.get("pdf_page_index", chunk_dict.get("page", 1)),
                printed_page_number=chunk_dict.get("printed_page_number"),
                section_heading=chunk_dict.get("section_heading"),
                text=chunk_dict.get("text", ""),
                page_char_start=chunk_dict.get("page_char_start", 0),
                page_char_end=chunk_dict.get("page_char_end", len(chunk_dict.get("text", ""))),
                score=float(score),
                warnings=list(chunk_dict.get("warnings", [])),
                extraction_method=chunk_dict.get("extraction_method", "native_text"),
            )
        )

    return results
