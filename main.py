#!/usr/bin/env python3
"""
main.py  –  Graph RAG Pipeline CLI

Usage examples
--------------
# Build graph (mock mode, no API key needed)
python main.py --chunks data/sample_chunks.json --mock

# Ask a natural-language graph query
python main.py --chunks data/sample_chunks.json --mock --query "products launched in 2020"
python main.py --chunks data/sample_chunks.json --mock --query "what services does the company offer"
python main.py --chunks data/sample_chunks.json --mock --query "acquisitions"
python main.py --chunks data/sample_chunks.json --mock --query "capabilities"
python main.py --chunks data/sample_chunks.json --mock --query "high confidence edges"

# Chunk retrieval (semantic search over raw text chunks)
python main.py --chunks data/sample_chunks.json --mock --retrieve "quick commerce products"

# LLM mode (requires Anthropic API key)
python main.py --chunks data/sample_chunks.json --api-key sk-ant-...

# Expand an existing graph with new chunks
python main.py --chunks data/new_chunks.json --mock --existing outputs/graph.json

# Verbose logging
python main.py --chunks data/sample_chunks.json --mock --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from graphrag.pipeline     import run_pipeline
from graphrag.query_engine import GraphQueryEngine



# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph RAG pipeline: chunks → knowledge graph + Mermaid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--chunks",      default="data/sample_chunks.json",
                   help="Path to JSON chunk file (default: data/sample_chunks.json)")
    p.add_argument("--mock",        action="store_true",
                   help="Use pre-extracted mock data — no API key required")
    p.add_argument("--api-key",     default=os.environ.get("ANTHROPIC_API_KEY"),
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    p.add_argument("--model",       default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                   help="Anthropic model identifier (default: claude-sonnet-5-5 or $ANTHROPIC_MODEL)")
    p.add_argument("--query",       default=None,
                   help="Natural-language graph query (e.g. 'products launched in 2020')")
    p.add_argument("--retrieve",    default=None,
                   help="Semantic chunk retrieval query (searches raw text chunks)")
    p.add_argument("--top-k",       type=int, default=3,
                   help="Number of chunks to return for --retrieve (default: 3)")
    p.add_argument("--existing",    default=None,
                   help="Path to existing graph JSON to expand with new chunks")
    p.add_argument("--output-dir",  default="outputs",
                   help="Directory for output files (default: outputs/)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose logging")
    return p.parse_args()


# ── Graph query dispatcher ────────────────────────────────────────────────────

def _run_graph_query(engine: GraphQueryEngine, query: str) -> None:
    """
    Route a natural-language --query to the appropriate query_engine method
    and pretty-print the results.
    """
    q = query.lower().strip()
    print(f'\n── Graph query: "{query}" {"─" * (44 - len(query))}')

    # ── Detect query intent ───────────────────────────────────────────────────

    # Products launched in a specific year
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", q)
    if ("launch" in q or "product" in q or "released" in q) and year_match:
        year = year_match.group(1)
        results = engine.find_products_launched(year=year)
        print(f"  Products / Features launched in {year}:\n")
        if results:
            for r in results:
                attrs = r["attributes"]
                detail = ""
                if "launch_year" in attrs:
                    detail += f", launch_year={attrs['launch_year']}"
                if "scale" in attrs:
                    detail += f", scale={attrs['scale']}"
                print(f"    • {r['name']} ({r['type']}{detail})  [conf={r['confidence']:.2f}]")
        else:
            print(f"    No results found for year {year}.")
        return

    # All products launched (no year filter)
    if "launch" in q or ("product" in q and "service" not in q):
        results = engine.find_products_launched()
        print("  All launched products / features:\n")
        for r in results:
            print(f"    • [{r['year']}]  {r['name']} ({r['type']})  [conf={r['confidence']:.2f}]")
        return

    # Services offered
    if "service" in q or "offer" in q or "provide" in q:
        results = engine.find_services_offered()
        print("  Services offered:\n")
        for r in results:
            print(f"    • {r['name']} (via {r['via']})")
        return

    # Acquisitions
    if "acqui" in q or "bought" in q or "purchased" in q or "merger" in q:
        results = engine.find_acquisitions()
        print("  Acquisitions:\n")
        for r in results:
            stake = f", stake={r['stake']}" if r['stake'] != 'unknown' else ""
            print(f"    • {r['name']}  acquired_year={r['acquired_year']}{stake}"
                  f"  [conf={r['confidence']:.2f}]")
        return

    # Capabilities
    if "capabilit" in q or "tech" in q or "analytics" in q or "fulfilment" in q:
        results = engine.find_capabilities()
        print("  Platform capabilities:\n")
        for r in results:
            print(f"    • {r['name']} ({r['type']})")
        return

    # Events by type
    for etype in ("launch", "acquisition", "milestone", "fundraise", "listing", "corporate"):
        if etype in q or ("event" in q and etype in q):
            results = engine.find_events_by_type(etype)
            print(f"  Events of type '{etype}':\n")
            for r in results:
                print(f"    • [{r['year']}]  {r['description'][:72]}")
            return

    # All events
    if "event" in q or "timeline" in q or "histor" in q or "year" in q:
        results = engine.find_events_by_type()
        print("  Full event timeline:\n")
        for r in results:
            print(f"    • [{r['year']}]  [{r['primary_event_type']}]  {r['description'][:65]}")
        return

    # Sources
    if "source" in q or "document" in q or "report" in q or "prospectus" in q:
        results = engine.find_sources()
        print("  Source documents:\n")
        for r in results:
            print(f"    • {r['source']}  ({r['count']} entities described)")
        return

    # High-confidence edges
    if "confidence" in q or "trust" in q or "reliable" in q or "high" in q:
        results = engine.filter_edges_by_confidence(0.85)
        print(f"  High-confidence edges (≥ 0.85):\n")
        for r in results[:15]:
            print(f"    • {r['from']}  →[{r['relation']}]→  {r['to']}"
                  f"  [conf={r['confidence']:.2f}, src={r['extraction_source']}]")
        if len(results) > 15:
            print(f"    … and {len(results) - 15} more")
        return

    # Partners
    if "partner" in q:
        results = engine.find_entities_by_type("Partner")
        print("  Partners:\n")
        for r in results:
            print(f"    • {r['name']}")
        return

    # Summary fallback
    print("  Showing graph summary (no specific pattern matched):\n")
    engine.print_summary()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if not args.mock and not args.api_key:
        print("ERROR: Pass --mock or supply an API key via --api-key / ANTHROPIC_API_KEY")
        sys.exit(1)

    # ── Load chunks ───────────────────────────────────────────────────────────
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"ERROR: Chunks file not found: {chunks_path}")
        sys.exit(1)
    with chunks_path.open() as f:
        chunks = json.load(f)

    # ── Load existing graph (optional) ────────────────────────────────────────
    existing_graph = None
    if args.existing:
        ep = Path(args.existing)
        if ep.exists():
            with ep.open() as f:
                existing_graph = json.load(f)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  Graph RAG Pipeline  v3")
    print("═" * 52)
    print(f"  Mode    : {'Mock' if args.mock else 'LLM (Anthropic API)'}")
    print(f"  Chunks  : {len(chunks)}")
    print(f"  Query   : {args.query or '(none)'}")
    print("═" * 52)

    result = run_pipeline(
        chunks,
        mock=args.mock,
        api_key=args.api_key,
        model=args.model,
        query=args.retrieve,
        top_k=args.top_k,
        existing_graph=existing_graph,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path   = out_dir / "graph.json"
    mermaid_path = out_dir / "graph.mmd"

    with graph_path.open("w") as f:
        json.dump(result["graph"], f, indent=2)
    with mermaid_path.open("w") as f:
        f.write(result["mermaid"])

    # ── Pretty graph summary ──────────────────────────────────────────────────
    engine = GraphQueryEngine(result["graph"])
    engine.print_summary()

    print(f"  Output files:")
    print(f"    {graph_path}")
    print(f"    {mermaid_path}")

    # ── Natural-language graph query (--query) ────────────────────────────────
    if args.query:
        _run_graph_query(engine, args.query)

    # ── Chunk retrieval (--retrieve) ──────────────────────────────────────────
    if args.retrieve and result["retrieval"]:
        print(f'\n── Chunk retrieval: "{args.retrieve}" {"─" * (40 - len(args.retrieve))}')
        for rank, (chunk, score) in enumerate(result["retrieval"], 1):
            snippet = chunk.get("text", "")[:110].replace("\n", " ")
            print(f"\n  #{rank}  Chunk {chunk.get('id','?')}  (score={score:.4f})")
            print(f"       {snippet}…")

    print()


if __name__ == "__main__":
    main()
