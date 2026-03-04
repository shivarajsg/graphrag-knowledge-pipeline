"""
graphrag.ingestion.schemas
--------------------------
Data structures and serialization schemas for document ingestion,
page extraction, chunking, and document indexing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractedTable:
    """Represents a structured table extracted from a document page."""
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    markdown: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedTable:
        bbox = tuple(data["bbox"]) if data.get("bbox") else None
        return cls(
            table_id=data.get("table_id", ""),
            headers=list(data.get("headers", [])),
            rows=[list(r) for r in data.get("rows", [])],
            bbox=bbox,
            markdown=data.get("markdown", ""),
        )


@dataclass
class ExtractedPage:
    """
    Represents an extracted physical page from a PDF document.
    Maintains physical page index (1-based), raw text, detected headings,
    tables, and extraction quality warnings.
    """
    pdf_page_index: int                       # 1-based physical PDF page number
    printed_page_number: Optional[str]        # Printed page label if detected; else None
    text: str                                 # Exact verbatim extracted text
    char_count: int = 0
    word_count: int = 0
    is_empty: bool = False
    is_suspicious_scanned: bool = False       # Flagged when low text with image content
    tables: List[ExtractedTable] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    extraction_method: str = "native_text"    # native_text | table | layout_blocks
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tables"] = [t.to_dict() for t in self.tables]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedPage:
        tables = [ExtractedTable.from_dict(t) for t in data.get("tables", [])]
        bbox = tuple(data["bounding_box"]) if data.get("bounding_box") else None
        return cls(
            pdf_page_index=data["pdf_page_index"],
            printed_page_number=data.get("printed_page_number"),
            text=data.get("text", ""),
            char_count=data.get("char_count", len(data.get("text", ""))),
            word_count=data.get("word_count", len(data.get("text", "").split())),
            is_empty=bool(data.get("is_empty", False)),
            is_suspicious_scanned=bool(data.get("is_suspicious_scanned", False)),
            tables=tables,
            headings=list(data.get("headings", [])),
            bounding_box=bbox,
            extraction_method=data.get("extraction_method", "native_text"),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class DocumentChunk:
    """
    Retrieval-friendly document chunk with complete source provenance.
    Strictly preserves physical PDF page number, exact text, character offsets,
    and extraction metadata.
    """
    chunk_id: str                             # Deterministic: {doc_id}_p{page:03d}_c{idx:02d}
    document_id: str                          # Stable document slug or hash
    source_file: str                          # Original filename (e.g. CITES_E-Appendices.pdf)
    document_hash: str                        # SHA-256 hash of original PDF content
    pdf_page_index: int                       # 1-based physical PDF page index
    printed_page_number: Optional[str]        # Printed page label if known, else None
    chunk_index: int                          # Position of chunk within this page (0-based)
    text: str                                 # Verbatim extracted source passage
    page_char_start: int                      # Start character index in extracted page.text
    page_char_end: int                        # End character index in extracted page.text
    section_heading: Optional[str] = None     # Detected section header or None
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    extraction_method: str = "native_text"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_retriever_chunk(self) -> Dict[str, Any]:
        """
        Export dictionary compatible with graphrag.retriever.retrieve_chunks.
        Carries all provenance fields alongside standard 'id', 'text', and 'page'.
        """
        d = self.to_dict()
        d["id"] = self.chunk_id
        d["page"] = self.pdf_page_index
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentChunk:
        bbox = tuple(data["bounding_box"]) if data.get("bounding_box") else None
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source_file=data["source_file"],
            document_hash=data["document_hash"],
            pdf_page_index=data["pdf_page_index"],
            printed_page_number=data.get("printed_page_number"),
            chunk_index=data.get("chunk_index", 0),
            text=data["text"],
            page_char_start=data.get("page_char_start", 0),
            page_char_end=data.get("page_char_end", len(data["text"])),
            section_heading=data.get("section_heading"),
            bounding_box=bbox,
            extraction_method=data.get("extraction_method", "native_text"),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class DocumentIndex:
    """
    Self-contained, serializable index of an ingested document.
    Includes document metadata, per-page extraction records, searchable chunks,
    and extraction warnings.
    """
    index_version: str = "1.0.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    document_id: str = ""
    source_file: str = ""
    document_hash: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    pages: List[ExtractedPage] = field(default_factory=list)
    chunks: List[DocumentChunk] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_version": self.index_version,
            "created_at": self.created_at,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "document_hash": self.document_hash,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "warnings": self.warnings,
            "config": self.config,
            "pages": [p.to_dict() for p in self.pages],
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def save(self, output_path: str | Path) -> None:
        """Serialize index to formatted JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentIndex:
        version = data.get("index_version", "1.0.0")
        if not version.startswith("1."):
            raise ValueError(f"Incompatible index_version '{version}'. Expected 1.x.x")
        pages = [ExtractedPage.from_dict(p) for p in data.get("pages", [])]
        chunks = [DocumentChunk.from_dict(c) for c in data.get("chunks", [])]
        return cls(
            index_version=version,
            created_at=data.get("created_at", ""),
            document_id=data.get("document_id", ""),
            source_file=data.get("source_file", ""),
            document_hash=data.get("document_hash", ""),
            total_pages=data.get("total_pages", len(pages)),
            total_chunks=data.get("total_chunks", len(chunks)),
            pages=pages,
            chunks=chunks,
            warnings=list(data.get("warnings", [])),
            config=dict(data.get("config", {})),
        )

    @classmethod
    def load(cls, index_path: str | Path) -> DocumentIndex:
        """Load and deserialize index from JSON file."""
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_retriever_chunks(self) -> List[Dict[str, Any]]:
        """Return list of chunk dicts formatted for graphrag.retriever.retrieve_chunks."""
        return [c.to_retriever_chunk() for c in self.chunks]
