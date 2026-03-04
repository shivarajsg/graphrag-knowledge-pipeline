"""
tests/test_ingestion.py
-----------------------
Automated test suite for Phase 1: PDF ingestion, layout extraction,
page-aware chunking, index serialization, and document search.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
import pytest

from graphrag.ingestion.chunker import DocumentChunker
from graphrag.ingestion.indexer import (
    index_pdf,
    search_document_index,
)
from graphrag.ingestion.pdf_loader import PDFLoader, compute_file_sha256
from graphrag.ingestion.schemas import (
    DocumentChunk,
    DocumentIndex,
    ExtractedPage,
    ExtractedTable,
)


# ── Helper to build controlled PDF fixtures ───────────────────────────────────

def create_sample_pdf(file_path: Path) -> Path:
    """
    Creates a realistic 3-page test PDF:
      Page 1: Title, Section heading, paragraphs, footer "Page 1".
      Page 2: Empty page (to verify empty page warnings).
      Page 3: Regulatory table and numbered clauses, footer "Page 3".
    """
    doc = fitz.open()

    # ── Page 1: Headings and paragraphs
    page1 = doc.new_page(width=595, height=842)  # A4
    page1.insert_text(
        (50, 50),
        "CONVENTION ON INTERNATIONAL TRADE IN ENDANGERED SPECIES\n\n"
        "FAUNA AND FLORA REGULATIONS\n\n"
        "1. GENERAL PROVISIONS\n"
        "This convention regulates international commercial trade in wild animals and plants.\n"
        "Specimens of species listed in Appendix I are subject to particularly strict regulation.\n"
        "Trade in specimens of these species must only be authorized in exceptional circumstances.\n",
        fontsize=11,
    )
    # Printed page footer at bottom margin
    page1.insert_text((280, 800), "Page 1", fontsize=10)

    # ── Page 2: Completely empty page
    page2 = doc.new_page(width=595, height=842)

    # ── Page 3: Tables and clauses
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text(
        (50, 50),
        "APPENDIX II SPECIES AND CONDITIONS\n\n"
        "Trade in specimens of Appendix II species requires the prior grant and presentation\n"
        "of an export permit. An export permit shall only be granted when an appropriate\n"
        "scientific authority has advised that such export will not be detrimental to the survival of that species.\n\n"
        "ORDER PRIMATES\n"
        "All species of Primates except those included in Appendix I are listed in Appendix II.\n",
        fontsize=11,
    )
    # Add a mock table drawing
    # In PyMuPDF, draw table borders to enable table recognition
    rect = fitz.Rect(50, 220, 540, 320)
    page3.draw_rect(rect, color=(0, 0, 0), width=1)
    # Draw header separator
    page3.draw_line(fitz.Point(50, 250), fitz.Point(540, 250), color=(0, 0, 0), width=1)
    # Draw vertical column line
    page3.draw_line(fitz.Point(250, 220), fitz.Point(250, 320), color=(0, 0, 0), width=1)
    page3.insert_text((60, 240), "Taxon", fontsize=10)
    page3.insert_text((260, 240), "Appendix Listing", fontsize=10)
    page3.insert_text((60, 280), "Macaca mulatta (Rhesus macaque)", fontsize=10)
    page3.insert_text((260, 280), "Appendix II", fontsize=10)

    page3.insert_text((280, 800), "Page 3", fontsize=10)

    doc.save(str(file_path))
    doc.close()
    return file_path


def create_scanned_mock_pdf(file_path: Path) -> Path:
    """Creates a 1-page PDF containing an image element and very little text (<50 chars)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Insert tiny 10-char text
    page.insert_text((50, 50), "Scanned doc", fontsize=10)
    # Insert a 1x1 dummy pixmap image
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), 1)
    pix.clear_with(255)
    page.insert_image(fitz.Rect(100, 100, 200, 200), pixmap=pix)
    doc.save(str(file_path))
    doc.close()
    return file_path


# ── PDF Ingestion Unit Tests ──────────────────────────────────────────────────

def test_pdf_loader_valid_document():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "sample.pdf"
        create_sample_pdf(pdf_path)

        loader = PDFLoader()
        result = loader.load(pdf_path)

        assert result["total_pages"] == 3
        assert result["source_file"] == "sample.pdf"
        assert len(result["document_hash"]) == 64
        assert result["document_id"].startswith("doc_")

        pages = result["pages"]
        assert len(pages) == 3

        # Page 1 checks
        p1 = pages[0]
        assert p1.pdf_page_index == 1
        assert p1.printed_page_number == "1"
        assert "ENDANGERED SPECIES" in p1.text
        assert not p1.is_empty
        assert any("FAUNA AND FLORA" in h or "PROVISIONS" in h for h in p1.headings)

        # Page 2 empty checks
        p2 = pages[1]
        assert p2.pdf_page_index == 2
        assert p2.is_empty
        assert any("empty" in w.lower() for w in p2.warnings)

        # Page 3 table / content checks
        p3 = pages[2]
        assert p3.pdf_page_index == 3
        assert p3.printed_page_number == "3"
        assert "PRIMATES" in p3.text
        assert "Macaca mulatta" in p3.text
    print("  ✓ test_pdf_loader_valid_document passed")


def test_pdf_loader_scanned_page_warning():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "scanned.pdf"
        create_scanned_mock_pdf(pdf_path)

        loader = PDFLoader(min_text_chars=50)
        result = loader.load(pdf_path)
        p = result["pages"][0]
        assert p.is_suspicious_scanned
        assert any("scanned" in w.lower() for w in p.warnings)
    print("  ✓ test_pdf_loader_scanned_page_warning passed")


def test_pdf_loader_invalid_file():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file_xyz.pdf")

    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = Path(tmpdir) / "empty.pdf"
        empty_file.write_bytes(b"")
        with pytest.raises(ValueError):
            loader.load(empty_file)

        corrupt_file = Path(tmpdir) / "corrupt.pdf"
        corrupt_file.write_bytes(b"Not a valid PDF header content")
        with pytest.raises(ValueError):
            loader.load(corrupt_file)
    print("  ✓ test_pdf_loader_invalid_file passed")


# ── Chunking Unit Tests ───────────────────────────────────────────────────────

def test_chunker_preserves_page_boundaries_and_offsets():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "sample.pdf"
        create_sample_pdf(pdf_path)

        loader = PDFLoader()
        result = loader.load(pdf_path)

        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document(
            document_id=result["document_id"],
            source_file=result["source_file"],
            document_hash=result["document_hash"],
            pages=result["pages"],
        )

        assert len(chunks) > 0

        # Verify no empty chunks
        for c in chunks:
            assert len(c.text.strip()) > 0
            assert c.page_char_end > c.page_char_start

        # Verify page boundary preservation: page 1 chunks have page index 1, etc.
        p1_chunks = [c for c in chunks if c.pdf_page_index == 1]
        p2_chunks = [c for c in chunks if c.pdf_page_index == 2]
        p3_chunks = [c for c in chunks if c.pdf_page_index == 3]

        assert len(p1_chunks) >= 1
        assert len(p2_chunks) == 0  # Page 2 was empty, must produce 0 chunks
        assert len(p3_chunks) >= 1

        # Check exact text offset fidelity for Page 1
        p1_text = result["pages"][0].text
        for c in p1_chunks:
            extracted_slice = p1_text[c.page_char_start:c.page_char_end]
            assert extracted_slice == c.text, "Chunk text must match exact page.text slice"

        # Check exact text offset fidelity for Page 3
        p3_text = result["pages"][2].text
        for c in p3_chunks:
            extracted_slice = p3_text[c.page_char_start:c.page_char_end]
            assert extracted_slice == c.text, "Chunk text must match exact page.text slice"
    print("  ✓ test_chunker_preserves_page_boundaries_and_offsets passed")


def test_chunk_id_stability_and_determinism():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "sample.pdf"
        create_sample_pdf(pdf_path)

        loader = PDFLoader()
        res1 = loader.load(pdf_path)
        res2 = loader.load(pdf_path)

        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks1 = chunker.chunk_document(res1["document_id"], res1["source_file"], res1["document_hash"], res1["pages"])
        chunks2 = chunker.chunk_document(res2["document_id"], res2["source_file"], res2["document_hash"], res2["pages"])

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
            assert c1.page_char_start == c2.page_char_start
            assert c1.page_char_end == c2.page_char_end
    print("  ✓ test_chunk_id_stability_and_determinism passed")


# ── Document Indexing & Search Unit Tests ─────────────────────────────────────

def test_document_index_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "sample.pdf"
        index_path = Path(tmpdir) / "index.json"
        create_sample_pdf(pdf_path)

        idx = index_pdf(pdf_path, chunk_size=400, chunk_overlap=50, output_path=index_path)

        assert index_path.exists()
        loaded_idx = DocumentIndex.load(index_path)

        assert loaded_idx.document_id == idx.document_id
        assert loaded_idx.document_hash == idx.document_hash
        assert loaded_idx.total_pages == idx.total_pages
        assert loaded_idx.total_chunks == idx.total_chunks
        assert len(loaded_idx.chunks) == len(idx.chunks)

        for orig_c, loaded_c in zip(idx.chunks, loaded_idx.chunks):
            assert orig_c.chunk_id == loaded_c.chunk_id
            assert orig_c.text == loaded_c.text
            assert orig_c.pdf_page_index == loaded_c.pdf_page_index
            assert orig_c.printed_page_number == loaded_c.printed_page_number
            assert orig_c.page_char_start == loaded_c.page_char_start
            assert orig_c.page_char_end == loaded_c.page_char_end
    print("  ✓ test_document_index_save_and_reload passed")


def test_search_document_index_retrieves_relevant_passages():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "sample.pdf"
        create_sample_pdf(pdf_path)

        idx = index_pdf(pdf_path, chunk_size=400, chunk_overlap=50)

        # Search for Macaca mulatta on page 3
        results = search_document_index(idx, query="Macaca mulatta Rhesus macaque", top_k=2)
        assert len(results) > 0
        top = results[0]
        assert "Macaca mulatta" in top.text
        assert top.pdf_page_index == 3
        assert top.printed_page_number == "3"
        assert top.source_file == "sample.pdf"
        assert top.score > 0.0

        # Search for general provisions on page 1
        res_p1 = search_document_index(idx, query="CONVENTION ON INTERNATIONAL TRADE", top_k=2)
        assert len(res_p1) > 0
        top_p1 = res_p1[0]
        assert "CONVENTION" in top_p1.text
        assert top_p1.pdf_page_index == 1
        assert top_p1.printed_page_number == "1"

        # Edge case: empty query returns empty list
        assert search_document_index(idx, query="", top_k=3) == []
    print("  ✓ test_search_document_index_retrieves_relevant_passages passed")


def test_ingestion_cli_end_to_end():
    import subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "cli_sample.pdf"
        index_path = Path(tmpdir) / "cli_index.json"
        create_sample_pdf(pdf_path)

        # 1. Ingest via CLI
        res_ingest = subprocess.run(
            [sys.executable, "-m", "graphrag.ingestion.cli", "--pdf", str(pdf_path), "--output", str(index_path)],
            capture_output=True,
            encoding="utf-8",
        )
        assert res_ingest.returncode == 0
        assert index_path.exists()
        assert "Ingestion Summary" in res_ingest.stdout

        # 2. Search via CLI
        res_search = subprocess.run(
            [sys.executable, "-m", "graphrag.ingestion.cli", "--index", str(index_path), "--query", "Macaca mulatta", "--top-k", "2"],
            capture_output=True,
            encoding="utf-8",
        )
        assert res_search.returncode == 0
        assert "Page 3" in res_search.stdout
        assert "Macaca mulatta" in res_search.stdout
    print("  ✓ test_ingestion_cli_end_to_end passed")


# ── CITES Integration Check (Runs only if CITES PDF is present) ───────────────

def test_cites_pdf_if_available():
    """
    Integration check: If CITES_E-Appendices.pdf is located in data/ or workspace,
    runs full ingestion and queries. If absent, logs informational notice.
    """
    candidates = [
        Path("CITES_E-Appendices.pdf"),
        Path("data/CITES_E-Appendices.pdf"),
        Path("data/test_documents/CITES_E-Appendices.pdf"),
    ]
    cites_path = next((p for p in candidates if p.exists()), None)
    if not cites_path:
        print("\n  [Notice] CITES_E-Appendices.pdf is not present in workspace. Skipping CITES file test.")
        return

    loader = PDFLoader()
    result = loader.load(cites_path)
    assert result["total_pages"] > 0
    print(f"\n  ✓ CITES PDF loaded successfully: {result['total_pages']} pages")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_ingestion_tests():
    tests = [
        test_pdf_loader_valid_document,
        test_pdf_loader_scanned_page_warning,
        test_pdf_loader_invalid_file,
        test_chunker_preserves_page_boundaries_and_offsets,
        test_chunk_id_stability_and_determinism,
        test_document_index_save_and_reload,
        test_search_document_index_retrieves_relevant_passages,
        test_ingestion_cli_end_to_end,
        test_cites_pdf_if_available,
    ]

    print("\n" + "=" * 55)
    print(" Graph RAG Ingestion (Phase 1) – Test Suite")
    print("=" * 55)

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 55)
    print(f" Ingestion Results: {passed} passed, {failed} failed")
    print("=" * 55 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_ingestion_tests()
    sys.exit(0 if success else 1)
