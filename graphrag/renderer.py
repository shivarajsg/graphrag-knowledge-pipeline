"""
renderer.py
-----------
Implements render_mermaid() – the third stage of the pipeline.

Converts a graph JSON (produced by build_graph) into a paste-ready
Mermaid `graph TD` string.

Features
--------
- Nodes grouped into labelled sections (Company, Services, Products, …)
- Rich labels: Type prefix + attributes formatted as key=value pairs
- Edges use controlled relation verbs as labels
- Event nodes get compact IDs (E2013, E2024, …)
- Source nodes connected via described_in edges are de-cluttered to
  only show connections from high-value (non-Event, non-Source) nodes
- Generated entirely from the graph JSON — no hand-writing
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .ontology import MERMAID_SECTION_ORDER
from .utils import sanitize_mermaid_id, escape_mermaid_label

# ── Label formatting ──────────────────────────────────────────────────────────

# Attributes to SKIP from label display (too verbose / internal)
_SKIP_ATTRS = {"description", "provenance", "tags"}

# Max number of attribute key=value pairs shown in a node label
_MAX_LABEL_ATTRS = 5


def _format_node_label(node: Dict[str, Any]) -> str:
    """
    Build a Mermaid-safe quoted label for a node.

    Format:
        "Type: Name<br/><b>Attributes</b>: key=value; …"

    For Event nodes the tags are shown instead of generic attributes.
    """
    ntype = node.get("type", "")
    label = node.get("label", node.get("id", ""))
    attrs = node.get("attributes", {})

    # ── Event nodes ────────────────────────────────────────────────────────
    if ntype == "Event":
        year = attrs.get("year", "")
        desc = attrs.get("description", "")
        tags = attrs.get("tags", [])
        tag_str = ", ".join(tags) if tags else ""
        parts = []
        if desc:
            # Truncate long descriptions
            short_desc = desc if len(desc) < 80 else desc[:77] + "…"
            parts.append(short_desc)
        if tag_str:
            parts.append(f"<b>Tags</b>: {tag_str}")
        inner = "<br/>".join(parts)
        full = f"Event: {label} ({year})"
        if inner:
            full += f"<br/>{inner}"
        return f'"{escape_mermaid_label(full)}"'

    # ── Source nodes ───────────────────────────────────────────────────────
    if ntype == "Source":
        return f'"Source: {escape_mermaid_label(label)}"'

    # ── Standard nodes ─────────────────────────────────────────────────────
    header = f"{ntype}: {label}"

    attr_pairs = []
    for k, v in attrs.items():
        if k in _SKIP_ATTRS:
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        attr_pairs.append(f"{k}={v}")
        if len(attr_pairs) >= _MAX_LABEL_ATTRS:
            break

    if attr_pairs:
        attr_str = "; ".join(attr_pairs)
        full_label = f"{header}<br/><b>Attributes</b>: {attr_str}"
    else:
        full_label = header

    return f'"{escape_mermaid_label(full_label)}"'


# ── Edge formatting ───────────────────────────────────────────────────────────

def _format_edge(edge: Dict[str, Any], mid_map: Dict[str, str]) -> str:
    """
    Return one Mermaid edge line:
        src_mid -->|relation| tgt_mid
    """
    src = mid_map.get(edge["source"], sanitize_mermaid_id(edge["source"]))
    tgt = mid_map.get(edge["target"], sanitize_mermaid_id(edge["target"]))
    rel = edge.get("relation", "related_to")
    return f"    {src} -->|{rel}| {tgt}"


# ── Public API ────────────────────────────────────────────────────────────────

def render_mermaid(graph: Dict[str, Any]) -> str:
    """
    Convert a graph JSON dict to a Mermaid `graph TD` string.

    Parameters
    ----------
    graph : dict produced by build_graph()

    Returns
    -------
    str – paste-ready Mermaid text
    """
    nodes: List[Dict] = graph.get("nodes", [])
    edges: List[Dict] = graph.get("edges", [])

    if not nodes:
        return "graph TD\n    %% Empty graph\n"

    # ── Build Mermaid ID map ────────────────────────────────────────────────
    # Event nodes → short compact ID (E2013, E2024, …)
    # All others  → sanitised slug of their graph ID
    mid_map: Dict[str, str] = {}
    for node in nodes:
        nid = node["id"]
        if node["type"] == "Event":
            year = node.get("attributes", {}).get("year", nid)
            mid_map[nid] = f"E{year}"
        else:
            mid_map[nid] = sanitize_mermaid_id(nid)

    # ── Bucket nodes by section ────────────────────────────────────────────
    type_to_section: Dict[str, str] = {}
    for section_name, types in MERMAID_SECTION_ORDER:
        for t in types:
            type_to_section[t] = section_name

    section_nodes: Dict[str, List[Dict]] = {
        s: [] for s, _ in MERMAID_SECTION_ORDER
    }
    for node in nodes:
        section = type_to_section.get(node["type"], "Other")
        section_nodes[section].append(node)

    # ── Separate described_in edges from structural edges ──────────────────
    # Only keep described_in from non-Event, non-Source, non-Unknown nodes
    # to avoid visual clutter.
    structural_edges = []
    source_edges = []
    for edge in edges:
        if edge["relation"] == "described_in":
            src_type = next(
                (n["type"] for n in nodes if n["id"] == edge["source"]), ""
            )
            if src_type not in ("Event", "Source", "Unknown"):
                source_edges.append(edge)
        else:
            structural_edges.append(edge)

    # ── Build Mermaid text ─────────────────────────────────────────────────
    lines: List[str] = ["graph TD"]

    for section_name, _ in MERMAID_SECTION_ORDER:
        section = section_nodes.get(section_name, [])
        if not section:
            continue

        lines.append("")
        lines.append("    %% " + "=" * 40)
        lines.append(f"    %% {section_name}")
        lines.append("    %% " + "=" * 40)

        for node in section:
            mid = mid_map[node["id"]]
            label = _format_node_label(node)
            lines.append(f"    {mid}[{label}]")

    # ── Structural edges ───────────────────────────────────────────────────
    lines.append("")
    lines.append("    %% " + "=" * 40)
    lines.append("    %% Relationships")
    lines.append("    %% " + "=" * 40)

    # Group edges by section for readability
    company_edges, event_edges, source_edge_lines, other_edges = [], [], [], []

    for edge in structural_edges:
        src_type = next(
            (n["type"] for n in nodes if n["id"] == edge["source"]), ""
        )
        if src_type == "Event":
            event_edges.append(edge)
        elif src_type == "Company" and edge["relation"] == "has_event":
            event_edges.append(edge)
        else:
            other_edges.append(edge)

    if other_edges:
        lines.append("")
        lines.append("    %% Core structure")
        for edge in other_edges:
            lines.append(_format_edge(edge, mid_map))

    if event_edges:
        lines.append("")
        lines.append("    %% Event timeline")
        for edge in event_edges:
            lines.append(_format_edge(edge, mid_map))

    if source_edges:
        lines.append("")
        lines.append("    %% Provenance (described_in)")
        for edge in source_edges:
            lines.append(_format_edge(edge, mid_map))

    return "\n".join(lines) + "\n"
