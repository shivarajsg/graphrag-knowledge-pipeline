"""
graphrag.ingestion.cli
----------------------
Command-line interface for PDF document ingestion, index generation,
and page-grounded passage search.

Usage Examples
--------------
# Build and save a document index from a PDF:
python -m graphrag.ingestion.cli --pdf data/test.pdf --output outputs/test_index.json

# Search an existing index:
python -m graphrag.ingestion.cli --index outputs/test_index.json --query "PRIMATES monkeys" --top-k 5

# Ingest and search in a single command:
python -m graphrag.ingestion.cli --pdf data/test.pdf --query "CITES Appendix" --top-k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import os
from .chunker import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from .indexer import index_pdf, search_document_index
from .schemas import DocumentIndex
from ..qa_engine import DocumentQAEngine, DEFAULT_QA_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m graphrag.ingestion.cli",
        description="PDF Ingestion, Document Search & Evidence-Grounded QA CLI (GraphRAG Phase 1 & 2)",
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to source PDF file to ingest",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the generated document index JSON",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to an existing document index JSON to load and search",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Search query to execute against the indexed document",
    )
    parser.add_argument(
        "--ask",
        "-a",
        type=str,
        default=None,
        help="Ask a natural-language question grounded in the indexed document (Phase 2)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock model for QA (no API key required)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key for live model generation (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_QA_MODEL,
        help=f"Anthropic model name (default: {DEFAULT_QA_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top relevant passages to return or use for QA (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk character overlap (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--table-lookup",
        type=str,
        default=None,
        help="Taxon or entity name to look up across tables with column-association safety (Phase 4)",
    )
    parser.add_argument(
        "--cross-refs",
        type=str,
        default=None,
        help="Chunk ID to scan and assemble multi-passage evidence chains (Phase 4)",
    )
    parser.add_argument(
        "--higher-taxon",
        type=str,
        default=None,
        help="Higher taxon premise to evaluate conditional taxonomic fallback (e.g. PRIMATES)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict document-only mode: reject external taxonomic premises not stated in the document",
    )
    return parser.parse_args()





def main() -> None:
    args = parse_args()

    if not args.pdf and not args.index:
        print("ERROR: You must specify either --pdf <file.pdf> or --index <index.json>.")
        sys.exit(1)

    index: DocumentIndex

    # ── Path A: Ingest new PDF ─────────────────────────────────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"ERROR: PDF file not found: {pdf_path}")
            sys.exit(1)

        print("\n" + "═" * 60)
        print("  PDF Document Ingestion Pipeline")
        print("═" * 60)
        print(f"  Source File   : {pdf_path.name}")
        print(f"  Chunk Size    : {args.chunk_size}")
        print(f"  Chunk Overlap : {args.chunk_overlap}")
        if args.output:
            print(f"  Output Index  : {args.output}")
        print("═" * 60)

        try:
            index = index_pdf(
                pdf_path=pdf_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                output_path=args.output,
            )
        except Exception as exc:
            print(f"\nERROR during PDF ingestion: {exc}")
            sys.exit(1)

        print("\n────────────────────────────────────────────────────────────")
        print("  Ingestion Summary")
        print("────────────────────────────────────────────────────────────")
        print(f"  Document ID   : {index.document_id}")
        print(f"  SHA-256 Hash  : {index.document_hash[:20]}…")
        print(f"  Total Pages   : {index.total_pages}")
        print(f"  Total Chunks  : {index.total_chunks}")

        # Page quality overview
        empty_pages = [p.pdf_page_index for p in index.pages if p.is_empty]
        scanned_pages = [p.pdf_page_index for p in index.pages if p.is_suspicious_scanned]
        table_pages = [p.pdf_page_index for p in index.pages if p.tables]

        if empty_pages:
            print(f"  Empty Pages   : {len(empty_pages)} (pages: {empty_pages[:10]}{'…' if len(empty_pages) > 10 else ''})")
        if scanned_pages:
            print(f"  Scanned Pages : {len(scanned_pages)} (pages: {scanned_pages[:10]}{'…' if len(scanned_pages) > 10 else ''})")
        if table_pages:
            total_tabs = sum(len(p.tables) for p in index.pages)
            print(f"  Tables Found  : {total_tabs} on {len(table_pages)} pages")

        if index.warnings:
            print(f"\n  Warnings ({len(index.warnings)}):")
            for w in index.warnings[:5]:
                print(f"    • {w}")
            if len(index.warnings) > 5:
                print(f"    … and {len(index.warnings) - 5} more")

        if args.output:
            print(f"\n  ✓ Index saved to: {args.output}")

    # ── Path B: Load existing index ───────────────────────────────────────────
    else:
        index_path = Path(args.index)
        if not index_path.exists():
            print(f"ERROR: Index file not found: {index_path}")
            sys.exit(1)

        print(f"\nLoading index from: {index_path} …")
        try:
            index = DocumentIndex.load(index_path)
            print(f"  Loaded document: {index.source_file} ({index.total_pages} pages, {index.total_chunks} chunks)")
        except Exception as exc:
            print(f"ERROR loading index: {exc}")
            sys.exit(1)

    # ── Search execution ──────────────────────────────────────────────────────
    if args.query:
        print("\n" + "═" * 60)
        print(f'  Document Search: "{args.query}"')
        print("═" * 60)

        results = search_document_index(index=index, query=args.query, top_k=args.top_k)

        if not results:
            print("  No matching passages found.")
        else:
            for rank, r in enumerate(results, 1):
                page_str = f"Page {r.pdf_page_index}"
                if r.printed_page_number:
                    page_str += f" (printed: {r.printed_page_number})"
                heading_str = f" | Section: {r.section_heading}" if r.section_heading else ""

                print(f"\n  [#{rank}] Score: {r.score:.4f}  │  {page_str}  │  ID: {r.chunk_id}{heading_str}")
                print(f"  {'─' * 56}")
                # Print clean snippet
                lines = [l.strip() for l in r.text.strip().splitlines() if l.strip()]
                preview = " ".join(lines)
                if len(preview) > 280:
                    preview = preview[:277] + "…"
                print(f"  \"{preview}\"")
                print(f"  [Offsets: chars {r.page_char_start}–{r.page_char_end}]")

    # ── Question Answering Execution (Phase 2) ───────────────────────────────
    if args.ask:
        print("\n" + "═" * 65)
        print("  Document Question Answering Engine (Phase 2)")
        print("═" * 65)
        print(f"  Question : {args.ask}")
        print(f"  Document : {index.source_file} (Pages: {index.total_pages})")
        print(f"  Mode     : {'Mock (Deterministic)' if args.mock else f'Live Anthropic ({args.model})'}")
        print("═" * 65)

        try:
            qa_engine = DocumentQAEngine(
                mock=args.mock,
                api_key=args.api_key,
                model=args.model,
                default_top_k=args.top_k,
            )
            response = qa_engine.answer_question(index=index, question=args.ask, top_k=args.top_k)
        except Exception as exc:
            print(f"\nERROR running Question Answering: {exc}")
            sys.exit(1)

        print(f"\n  Status: [{response.status.upper()}]")
        print("─" * 65)
        print("  ANSWER:")
        print(f"  {response.answer}")

        if response.claims:
            print("\n  EVALUATED CLAIMS & VERIFICATION STATUS:")
            for idx, c in enumerate(response.claims, 1):
                inf_label = " [Inference]" if c.is_inference else ""
                eids_str = ", ".join(c.evidence_ids) if c.evidence_ids else "None"
                cid_label = f"[{c.claim_id or f'C{idx}'}]"
                v_status = c.verification_status.upper() if getattr(c, "verification_status", None) else "VERIFIED"
                print(f"    {cid_label} {c.claim}{inf_label} (Cited Evidence: {eids_str})")
                print(f"         Status : [{v_status}] via {getattr(c, 'verification_method', 'citation')}")
                if getattr(c, "explanation", None):
                    print(f"         Details: {c.explanation}")
                if getattr(c, "contradiction_details", None):
                    print(f"         Conflict: {c.contradiction_details}")


        if response.evidence:
            print("\n  SOURCE CITATIONS & PASSAGES:")
            for ev in response.evidence:
                page_str = f"Page {ev.pdf_page_index}"
                if ev.printed_page_number:
                    page_str += f" (printed: {ev.printed_page_number})"
                heading_str = f" | Section: {ev.section_heading}" if ev.section_heading else ""
                print(f"\n    [{ev.evidence_id}] {ev.source_file} │ {page_str} │ Chunk: {ev.chunk_id}{heading_str}")
                print(f"        \"{ev.quote}\"")
                if ev.page_char_start is not None and ev.page_char_end is not None:
                    print(f"        [Page char offsets: {ev.page_char_start}–{ev.page_char_end}]")

        if response.evidence_trail:
            print("\n  EVIDENCE TRAIL:")
            for step in response.evidence_trail:
                print(f"    • {step}")

        if response.limitations:
            print("\n  LIMITATIONS & UNADDRESSED ASPECTS:")
            for lim in response.limitations:
                print(f"    ⚠ {lim}")

        if response.validation_warnings:
            print("\n  VALIDATION WARNINGS:")
            for w in response.validation_warnings:
                print(f"    ! {w}")

        print("\n" + "═" * 65)

    # ── Table Reasoner Execution (Phase 4) ───────────────────────────────────
    if args.table_lookup:
        from ..rule_engine import TableReasoner
        print("\n" + "═" * 65)
        print("  Table Reasoner: Taxonomic & Column Classification (Phase 4)")
        print("═" * 65)
        print(f"  Target Taxon : {args.table_lookup}")
        if args.higher_taxon:
            print(f"  Higher Taxon : {args.higher_taxon} (External Premise)")
        print(f"  Policy       : {'Strict Document-Only' if args.strict else 'Standard'}")
        print("═" * 65)

        reasoner = TableReasoner(index)
        res = reasoner.resolve_cites_classification(
            species_name=args.table_lookup,
            higher_taxon=args.higher_taxon,
            strict_document_only=args.strict,
        )

        status_label = {
            "direct_listing": "DIRECT DOCUMENT LISTING",
            "conditional_inference": "CONDITIONAL TAXONOMIC INFERENCE (NOT IN DOCUMENT VERBATIM)",
            "unlisted_in_document": "UNLISTED IN DOCUMENT / INSUFFICIENT EVIDENCE",
            "unsupported_taxonomic_mapping": "UNSUPPORTED TAXONOMIC MAPPING",
        }.get(res.get("status", ""), res.get("status", "").upper())

        print(f"\n  Status         : [{status_label}]")
        print(f"  Classification : {res['classification']}")
        print(f"  Rationale      : {res['rationale']}")

        if res.get("assumptions"):
            print("\n  ASSUMPTIONS / EXTERNAL PREMISES:")
            for a in res["assumptions"]:
                print(f"    ⚠ {a}")

        if res.get("provenance"):
            prov = res["provenance"]
            print(f"\n  CITATION & PROVENANCE:")
            print(f"    Page       : {prov['pdf_page_index']}")
            print(f"    Column     : {prov['column_heading']}")
            print(f"    Table ID   : {prov['table_id']} (Row {prov['row_index']})")
            if res.get("verbatim_source_entry"):
                print(f"    Cited Text : \"{res['verbatim_source_entry']}\"")
            if prov.get("modifying_footnotes"):
                print(f"    Footnotes  : {', '.join(prov['modifying_footnotes'])}")
        print("\n" + "═" * 65)


    # ── Multi-Passage Cross-References Execution (Phase 4) ────────────────────
    if args.cross_refs:
        from ..rule_engine import MultiPassageAssembler
        print("\n" + "═" * 65)
        print("  Multi-Passage Evidence Assembler (Phase 4)")
        print("═" * 65)
        print(f"  Root Chunk ID: {args.cross_refs}")
        print("═" * 65)
        assembler = MultiPassageAssembler(index)
        evidence = assembler.assemble_multi_passage_evidence(args.cross_refs)
        print(f"  Connected Chunks    : {len(evidence.connected_chunk_ids)} ({', '.join(evidence.connected_chunk_ids) or 'None'})")
        print(f"  Cross-References    : {', '.join(evidence.cross_references_resolved) or 'None'}")
        print(f"  Unresolved Pointers : {', '.join(evidence.unresolved_references) or 'None'}")
        print(f"  Footnotes Detected  : {', '.join(evidence.footnotes) or 'None'}")
        if evidence.combined_context:
            preview = evidence.combined_context[:300].strip()
            print(f"\n  COMBINED CONTEXT PREVIEW:\n  \"{preview}…\"")
        print("\n" + "═" * 65)

    print()




if __name__ == "__main__":
    main()
