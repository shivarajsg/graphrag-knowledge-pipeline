"""
pipeline.py
-----------
Orchestrates the full Graph RAG pipeline:

  chunks → extract() → build_graph() → render_mermaid()
                                     ↘ retrieve_chunks()  (optional)

Public function
---------------
  run_pipeline(chunks, *, mock, api_key, query, top_k, existing_graph)
      → {"graph": {...}, "mermaid": "...", "retrieval": [...]}

This is the single entry point used by main.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .extractor import extract
from .graph_builder import build_graph
from .renderer import render_mermaid
from .retriever import retrieve_chunks as _retrieve

logger = logging.getLogger(__name__)


def run_pipeline(
    chunks: List[Dict[str, Any]],
    *,
    mock: bool = True,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    query: Optional[str] = None,
    top_k: int = 3,
    existing_graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full Graph RAG pipeline.

    Parameters
    ----------
    chunks         : list of chunk dicts (must have 'id' and 'text')
    mock           : use pre-extracted data instead of calling the LLM
    api_key        : Anthropic API key (required when mock=False)
    model          : Anthropic model to use (optional, forwarded to extract)
    query          : optional retrieval query
    top_k          : number of chunks to return for retrieval
    existing_graph : an existing graph JSON to expand (pass None for fresh)

    Returns
    -------
    {
        "extractions" : list of per-chunk extraction dicts,
        "graph"       : graph JSON (nodes + edges + metadata),
        "mermaid"     : paste-ready Mermaid string,
        "retrieval"   : list of (chunk, score) tuples (empty if no query),
    }
    """
    # ── Stage 1: Extract ─────────────────────────────────────────────────────
    logger.info(f"[pipeline] Stage 1 – Extracting from {len(chunks)} chunk(s) …")
    extractions: List[Dict] = []
    for chunk in chunks:
        extraction = extract(chunk, mock=mock, api_key=api_key, model=model)
        extractions.append(extraction)
        logger.info(
            f"  chunk {chunk.get('id','?')}: "
            f"{len(extraction['entities'])} entities, "
            f"{len(extraction['relations'])} relations, "
            f"{len(extraction['events'])} events"
        )

    # ── Stage 2: Build graph ──────────────────────────────────────────────────
    logger.info("[pipeline] Stage 2 – Building graph …")
    graph = build_graph(extractions, existing_graph=existing_graph)
    logger.info(
        f"[pipeline] Graph: {graph['metadata']['node_count']} nodes, "
        f"{graph['metadata']['edge_count']} edges"
    )

    # ── Stage 3: Render Mermaid ───────────────────────────────────────────────
    logger.info("[pipeline] Stage 3 – Rendering Mermaid …")
    mermaid_text = render_mermaid(graph)

    # ── Stage 4 (optional): Retrieve relevant chunks ──────────────────────────
    retrieval = []
    if query:
        logger.info(f"[pipeline] Stage 4 – Retrieving chunks for: {query!r}")
        retrieval = _retrieve(query, chunks, top_k=top_k)
        logger.info(
            f"[pipeline] Top {len(retrieval)} result(s): "
            + ", ".join(f"chunk {c.get('id','?')} ({s:.3f})" for c, s in retrieval)
        )

    return {
        "extractions": extractions,
        "graph": graph,
        "mermaid": mermaid_text,
        "retrieval": retrieval,
    }
