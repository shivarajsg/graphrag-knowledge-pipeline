"""
graphrag.ingestion.chunker
--------------------------
Page-aware text chunking component.
Divides extracted PDF pages into retrieval-friendly passages while
preserving exact source text, physical page index, character offsets,
and section headings.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .schemas import DocumentChunk, ExtractedPage

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


class DocumentChunker:
    """
    Page-aware chunker that splits pages into retrieval chunks without
    crossing page boundaries, preserving character offsets and exact quotes.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document_id: str,
        source_file: str,
        document_hash: str,
        pages: List[ExtractedPage],
    ) -> List[DocumentChunk]:
        """
        Divide all pages of a document into DocumentChunk objects.

        Parameters
        ----------
        document_id   : stable document identifier
        source_file   : original filename
        document_hash : SHA-256 hash of document
        pages         : list of ExtractedPage objects

        Returns
        -------
        List of DocumentChunk instances
        """
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            page_chunks = self.chunk_page(
                document_id=document_id,
                source_file=source_file,
                document_hash=document_hash,
                page=page,
            )
            all_chunks.extend(page_chunks)

        logger.info(
            f"[DocumentChunker] Chunked '{source_file}' ({len(pages)} pages) → "
            f"{len(all_chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def chunk_page(
        self,
        document_id: str,
        source_file: str,
        document_hash: str,
        page: ExtractedPage,
    ) -> List[DocumentChunk]:
        """
        Split a single ExtractedPage into chunks.
        Never combines text across multiple pages.
        """
        raw_text = page.text
        if not raw_text or not raw_text.strip():
            return []

        chunks: List[DocumentChunk] = []

        # Find paragraphs and their character spans in page.text
        para_spans = self._split_into_paragraph_spans(raw_text)
        if not para_spans:
            return []

        current_start = 0
        current_end = 0
        current_spans: List[tuple[int, int]] = []
        chunk_idx = 0

        # Closest heading tracker
        active_heading = page.headings[0] if page.headings else None

        for start_idx, end_idx in para_spans:
            para_text = raw_text[start_idx:end_idx].strip()
            # Detect if this paragraph starts with one of the page's section headings
            starts_heading = any(para_text.startswith(h) or h in para_text.splitlines()[:2] for h in page.headings) if page.headings else False

            # Check if adding this paragraph exceeds chunk_size OR if a distinct section heading begins
            potential_length = (end_idx - current_spans[0][0]) if current_spans else (end_idx - start_idx)

            if current_spans and (potential_length > self.chunk_size or (starts_heading and potential_length > 150)):
                # Emit current accumulated chunk
                chunk_start = current_spans[0][0]
                chunk_end = current_spans[-1][1]
                chunk_text = raw_text[chunk_start:chunk_end]

                chunk_id = f"{document_id}_p{page.pdf_page_index:03d}_c{chunk_idx:02d}"
                heading = self._find_heading_for_span(page.headings, chunk_text, active_heading)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source_file=source_file,
                        document_hash=document_hash,
                        pdf_page_index=page.pdf_page_index,
                        printed_page_number=page.printed_page_number,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        page_char_start=chunk_start,
                        page_char_end=chunk_end,
                        section_heading=heading,
                        bounding_box=page.bounding_box,
                        extraction_method=page.extraction_method,
                        warnings=list(page.warnings),
                    )
                )
                chunk_idx += 1
                active_heading = heading or active_heading

                # Retain overlap spans if possible
                overlap_target = chunk_end - self.chunk_overlap
                retained_spans = [s for s in current_spans if s[1] > overlap_target]
                current_spans = retained_spans

            # If a single paragraph is longer than chunk_size, split by sentences/lines
            if (end_idx - start_idx) > self.chunk_size:
                sub_spans = self._split_long_paragraph(raw_text, start_idx, end_idx)
                for sub_s, sub_e in sub_spans:
                    sub_text = raw_text[sub_s:sub_e]
                    if not sub_text.strip():
                        continue
                    chunk_id = f"{document_id}_p{page.pdf_page_index:03d}_c{chunk_idx:02d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            source_file=source_file,
                            document_hash=document_hash,
                            pdf_page_index=page.pdf_page_index,
                            printed_page_number=page.printed_page_number,
                            chunk_index=chunk_idx,
                            text=sub_text,
                            page_char_start=sub_s,
                            page_char_end=sub_e,
                            section_heading=self._find_heading_for_span(page.headings, sub_text, active_heading),
                            bounding_box=page.bounding_box,
                            extraction_method=page.extraction_method,
                            warnings=list(page.warnings),
                        )
                    )
                    chunk_idx += 1
                current_spans = []
            else:
                current_spans.append((start_idx, end_idx))

        # Emit remaining spans
        if current_spans:
            chunk_start = current_spans[0][0]
            chunk_end = current_spans[-1][1]
            chunk_text = raw_text[chunk_start:chunk_end]
            if chunk_text.strip():
                chunk_id = f"{document_id}_p{page.pdf_page_index:03d}_c{chunk_idx:02d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source_file=source_file,
                        document_hash=document_hash,
                        pdf_page_index=page.pdf_page_index,
                        printed_page_number=page.printed_page_number,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        page_char_start=chunk_start,
                        page_char_end=chunk_end,
                        section_heading=self._find_heading_for_span(page.headings, chunk_text, active_heading),
                        bounding_box=page.bounding_box,
                        extraction_method=page.extraction_method,
                        warnings=list(page.warnings),
                    )
                )

        return chunks

    def _split_into_paragraph_spans(self, text: str) -> List[tuple[int, int]]:
        """Find non-empty paragraph character spans separated by double newlines."""
        spans: List[tuple[int, int]] = []
        last = 0
        for m in re.finditer(r"\n\s*\n+", text):
            s, e = last, m.start()
            # Trim whitespace from boundaries
            while e > s and text[e - 1].isspace():
                e -= 1
            while s < e and text[s].isspace():
                s += 1
            if s < e:
                spans.append((s, e))
            last = m.end()

        if last < len(text):
            s, e = last, len(text)
            while e > s and text[e - 1].isspace():
                e -= 1
            while s < e and text[s].isspace():
                s += 1
            if s < e:
                spans.append((s, e))
        return spans

    def _split_long_paragraph(self, text: str, start: int, end: int) -> List[tuple[int, int]]:
        """Split a long paragraph exceeding chunk_size into smaller sentence/line spans."""
        spans: List[tuple[int, int]] = []
        cur_start = start
        while cur_start < end:
            target_end = min(cur_start + self.chunk_size, end)
            if target_end < end:
                # Look for sentence boundary or newline before target_end
                slice_text = text[cur_start:target_end]
                split_point = -1
                for sep in [". ", "?\n", "!\n", "\n", "; "]:
                    pos = slice_text.rfind(sep)
                    if pos > self.chunk_overlap:
                        split_point = cur_start + pos + len(sep)
                        break
                if split_point > cur_start:
                    target_end = split_point

            spans.append((cur_start, target_end))
            # Advance with overlap
            next_start = target_end - self.chunk_overlap
            if next_start <= cur_start or next_start >= end:
                next_start = target_end
            cur_start = next_start
        return spans

    def _find_heading_for_span(
        self, headings: List[str], text: str, fallback_heading: Optional[str]
    ) -> Optional[str]:
        """Detect if chunk contains or is immediately preceded by a section heading."""
        if not headings:
            return fallback_heading
        # Look for headings contained in this chunk; pick the one appearing in this chunk's text
        matches = []
        for h in headings:
            pos = text.find(h)
            if pos != -1:
                matches.append((pos, h))
        if matches:
            # Sort by position in chunk text; pick the earliest heading starting in this chunk
            matches.sort(key=lambda x: x[0])
            return matches[0][1]
        return fallback_heading
