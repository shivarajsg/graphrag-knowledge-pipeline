"""
graphrag.ingestion
------------------
Document ingestion, PDF parsing, page-aware chunking, and index management
for verifiable, grounded document retrieval.
"""

from .chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DocumentChunker
from .indexer import DocumentSearchResult, index_pdf, search_document_index
from .pdf_loader import PDFLoader, compute_file_sha256
from .schemas import (
    DocumentChunk,
    DocumentIndex,
    ExtractedPage,
    ExtractedTable,
)

__all__ = [
    "PDFLoader",
    "compute_file_sha256",
    "DocumentChunker",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "ExtractedTable",
    "ExtractedPage",
    "DocumentChunk",
    "DocumentIndex",
    "index_pdf",
    "search_document_index",
    "DocumentSearchResult",
]
