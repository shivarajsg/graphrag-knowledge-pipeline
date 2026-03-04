"""
graphrag.ingestion.pdf_loader
-----------------------------
Robust, page-aware PDF text extraction and layout parsing using PyMuPDF (fitz).
Preserves physical page indices, table structures, headings, and flags
empty or potentially scanned pages with clear warnings.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import ExtractedPage, ExtractedTable

logger = logging.getLogger(__name__)

# Fallback threshold: pages with fewer characters than this are inspected for scanned images
DEFAULT_MIN_TEXT_CHARS = 50

# Regex patterns for detecting printed page numbers in header/footer margins
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*page\s+(\d+)\b", re.I),
    re.compile(r"^\s*-\s*(\d+)\s*-\s*$"),
    re.compile(r"^\s*(\d+)\s+of\s+\d+\s*$", re.I),
    re.compile(r"^\s*(\d+)\s*$"),
]


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class PDFLoader:
    """
    Page-aware PDF document loader.
    Extracts text, table structures, headings, and flags extraction deficiencies.
    """

    def __init__(self, min_text_chars: int = DEFAULT_MIN_TEXT_CHARS) -> None:
        self.min_text_chars = min_text_chars

    def load(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Load and parse a PDF file into structured page records.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        Dict with document-level metadata and list of ExtractedPage objects.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty (0 bytes): {path}")

        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for PDF ingestion. Install with: pip install pymupdf"
            ) from e

        doc_hash = compute_file_sha256(path)
        doc_id = f"doc_{doc_hash[:12]}"
        doc_warnings: List[str] = []

        doc = None
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Failed to open PDF '{path.name}': {exc}") from exc

        try:
            if doc.is_encrypted:
                doc_warnings.append("Document is encrypted.")
                if not doc.authenticate(""):
                    raise ValueError(f"PDF is password-protected and could not be unlocked: {path.name}")

            page_count = len(doc)
            if page_count == 0:
                doc_warnings.append("PDF contains 0 pages.")

            extracted_pages: List[ExtractedPage] = []

            for page_idx in range(page_count):
                page = doc[page_idx]
                physical_page_num = page_idx + 1  # 1-based index
                page_warnings: List[str] = []

                # Page dimensions
                rect = page.rect
                bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))

                # Raw verbatim text
                raw_text = page.get_text("text")

                # Layout blocks: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type == 0 is text, block_type == 1 is image
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip() and (len(b) < 7 or b[6] == 0)]

                # Detect printed page number from margin blocks
                printed_page = self._detect_printed_page(text_blocks, rect.height)

                # Detect section headings
                headings = self._detect_headings(text_blocks)

                # Detect and extract structured tables
                tables = self._extract_tables(page, physical_page_num)

                # Quality checks
                char_count = len(raw_text.strip())
                is_empty = char_count == 0
                image_count = len(page.get_images())
                is_scanned = (char_count < self.min_text_chars) and (image_count > 0)

                if is_empty:
                    page_warnings.append(f"Page {physical_page_num} is empty: no text extracted.")
                elif is_scanned:
                    page_warnings.append(
                        f"Page {physical_page_num} is potentially scanned: only {char_count} chars extracted "
                        f"with {image_count} embedded image(s)."
                    )
                elif char_count < self.min_text_chars:
                    page_warnings.append(
                        f"Page {physical_page_num} has unusually short text ({char_count} chars)."
                    )

                if tables:
                    page_warnings.append(f"Page {physical_page_num} contains {len(tables)} detected table(s).")

                extraction_method = "table" if tables else ("layout_blocks" if text_blocks else "native_text")

                extracted_pages.append(
                    ExtractedPage(
                        pdf_page_index=physical_page_num,
                        printed_page_number=printed_page,
                        text=raw_text,
                        char_count=len(raw_text),
                        word_count=len(raw_text.split()),
                        is_empty=is_empty,
                        is_suspicious_scanned=is_scanned,
                        tables=tables,
                        headings=headings,
                        bounding_box=bbox,
                        extraction_method=extraction_method,
                        warnings=page_warnings,
                    )
                )

        finally:
            if doc is not None:
                doc.close()

        logger.info(
            f"[PDFLoader] Processed '{path.name}': {len(extracted_pages)} pages, "
            f"doc_id={doc_id}"
        )

        return {
            "document_id": doc_id,
            "source_file": path.name,
            "document_hash": doc_hash,
            "total_pages": len(extracted_pages),
            "pages": extracted_pages,
            "warnings": doc_warnings,
        }

    def _detect_printed_page(self, blocks: List[Tuple], page_height: float) -> Optional[str]:
        """
        Inspect header (top 10%) and footer (bottom 10%) margin blocks to identify
        a printed page number. Never fabricates a number if ambiguous.
        """
        top_margin = page_height * 0.10
        bottom_margin = page_height * 0.90

        candidate_lines: List[str] = []
        for b in blocks:
            if len(b) < 5:
                continue
            y0, y1, btext = b[1], b[3], b[4].strip()
            if y0 <= top_margin or y1 >= bottom_margin:
                lines = [l.strip() for l in btext.splitlines() if l.strip()]
                candidate_lines.extend(lines)

        for line in candidate_lines:
            for pattern in _PAGE_NUM_PATTERNS:
                m = pattern.match(line)
                if m:
                    page_val = m.group(1)
                    # Validate that it is a plausible numeric page string
                    if page_val.isdigit() and 1 <= int(page_val) <= 9999:
                        return page_val
        return None

    def _detect_headings(self, blocks: List[Tuple]) -> List[str]:
        """
        Extract prominent section or division headings based on uppercase patterns,
        short header lines, taxonomic order titles, or legal division keywords (e.g. FAUNA, MAMMALIA, PRIMATES).
        Excludes parenthetical exception clauses and inline annotations.
        """
        headings: List[str] = []
        heading_keywords = {"APPENDIX", "ANNEX", "CHAPTER", "ARTICLE", "SECTION", "PART", "SCHEDULE"}
        neg_phrases = {
            "included in", "except the species", "see appendix", "population of",
            "zero export quota", "specimens of", "subject to", "excluding"
        }
        stop_words = {"SPP", "AND", "THE", "FOR", "WITH", "FROM", "UNDER", "PAGE", "TOTAL"}

        for b in blocks:
            if len(b) < 5:
                continue
            lines = [l.strip() for l in b[4].splitlines() if l.strip()]
            for line in lines:
                # Heading heuristic: clean line <= 80 chars
                if len(line) <= 80:
                    # Ignore lines starting with parenthesis or containing exception clauses
                    if line.startswith("(") or line.startswith("[") or line.endswith(")"):
                        continue
                    line_lower = line.lower()
                    if any(neg in line_lower for neg in neg_phrases):
                        continue

                    upper_line = line.upper()
                    tokens = line.split()
                    first_token = tokens[0].strip(":,.-") if tokens else ""

                    # 1. Major division keyword match (e.g. "Appendices I, II & III", "SECTION 1")
                    is_kw_match = any(upper_line.startswith(kw) or f" {kw} " in f" {upper_line} " for kw in heading_keywords)

                    # 2. Complete uppercase line (e.g. "PHOLIDOTA", "MAMMALIA")
                    is_all_caps = (line == upper_line) and any(c.isalpha() for c in line) and len(line) > 3

                    # 3. Taxonomic order heading (e.g. "PRIMATES Apes, monkeys")
                    is_taxonomic_order = (
                        len(first_token) >= 4
                        and first_token.isupper()
                        and first_token.isalpha()
                        and first_token not in stop_words
                    )

                    if is_kw_match or is_all_caps or is_taxonomic_order:
                        heading_to_add = first_token if (is_taxonomic_order and not is_kw_match and not is_all_caps) else line
                        if heading_to_add not in headings:
                            headings.append(heading_to_add)
                            if len(headings) >= 15:
                                return headings
        return headings

    def _extract_tables(self, page: Any, page_num: int) -> List[ExtractedTable]:
        """
        Attempt structured table extraction using PyMuPDF's page.find_tables().
        Safely falls back if the method is not supported or finds no tables.
        """
        tables: List[ExtractedTable] = []
        if not hasattr(page, "find_tables"):
            return tables

        try:
            tabs = page.find_tables()
            for t_idx, tab in enumerate(tabs):
                df_rows = tab.extract()
                if not df_rows or len(df_rows) < 2:
                    continue

                headers = [str(c or "").strip() for c in df_rows[0]]
                data_rows = [[str(c or "").strip() for c in row] for row in df_rows[1:]]

                # Format clean markdown table
                md_lines = []
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for r in data_rows:
                    md_lines.append("| " + " | ".join(r) + " |")
                md_table = "\n".join(md_lines)

                bbox = None
                if hasattr(tab, "bbox"):
                    bbox = (float(tab.bbox[0]), float(tab.bbox[1]), float(tab.bbox[2]), float(tab.bbox[3]))

                tables.append(
                    ExtractedTable(
                        table_id=f"p{page_num:03d}_t{t_idx+1:02d}",
                        headers=headers,
                        rows=data_rows,
                        bbox=bbox,
                        markdown=md_table,
                    )
                )
        except Exception as e:
            logger.debug(f"[pdf_loader] Table extraction warning on page {page_num}: {e}")

        return tables
