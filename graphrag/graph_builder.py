"""
graph_builder.py
----------------
Implements build_graph() – the second stage of the pipeline.

Takes a list of extraction dicts (from extractor.extract()) and builds
(or expands) a Graph RAG-ready JSON structure:

  {
    "nodes": [ {id, type, label, attributes, provenance}, … ],
    "edges": [ {source, target, relation, raw_relation, provenance}, … ],
    "metadata": { created_at, chunk_count, node_count, edge_count }
  }

Key design decisions
--------------------
- Stable IDs: derived from (normalised name + type) so they survive
  across multiple chunk batches.
- Entity resolution: 'Swiggy' and 'Swiggy Limited' collapse to the
  same node via normalize_name().
- Expandability: pass an existing graph dict as `existing_graph` to
  add new nodes/edges without duplication.
- Event handling: one event node per (company, year); multiple events
  in the same year are merged into one node with a combined description.
- Provenance: every node and edge carries the list of chunk_ids that
  contributed to it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .ontology import NODE_TYPES, CONFIDENCE_SCORES
from .utils import (
    make_node_id,
    make_event_id,
    normalize_name,
    fuzzy_match_entity,
    map_relation,
    merge_attributes,
    merge_provenance,
    dedup_nodes,
    dedup_edges,
    extract_numeric_attributes,
    utc_now,
)

logger = logging.getLogger(__name__)


def _classify_event_types(tags: List[str]) -> List[str]:
    """
    Map ontology tags to structured event_type values.
    Returns a list of one or more event type strings.
    """
    TAG_TO_TYPE = {
        "Launch":      "launch",
        "Pioneer":     "launch",
        "Acquisition": "acquisition",
        "Fundraise":   "fundraise",
        "Listing":     "listing",
        "Milestone":   "milestone",
        "Scale":       "milestone",
        "Corporate":   "corporate",
    }
    types = []
    for tag in tags:
        et = TAG_TO_TYPE.get(tag)
        if et and et not in types:
            types.append(et)
    return types if types else ["other"]


# ── Public API ────────────────────────────────────────────────────────────────

def build_graph(
    extractions: List[Dict[str, Any]],
    existing_graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build or expand a knowledge-graph JSON from extraction results.

    Parameters
    ----------
    extractions    : list of dicts returned by extractor.extract()
    existing_graph : optional previous graph to expand (pass None for fresh build)

    Returns
    -------
    Graph dict with 'nodes', 'edges', 'metadata' keys.
    """
    # ── Seed from existing graph (if expanding) ──────────────────────────────
    nodes: Dict[str, Dict] = {}   # id → node dict
    edges: List[Dict] = []

    if existing_graph:
        for n in existing_graph.get("nodes", []):
            nodes[n["id"]] = n
        edges = list(existing_graph.get("edges", []))

    # ── Label index for fuzzy resolution ────────────────────────────────────
    # Maps normalized_label → node_id for quick lookup
    def _label_index() -> Dict[str, str]:
        return {normalize_name(n["label"]): n["id"] for n in nodes.values()}

    # ── Helper: upsert a node ────────────────────────────────────────────────
    def upsert_node(
        name: str,
        node_type: str,
        attributes: Dict,
        chunk_id: str,
        type_method: str = "llm",
    ) -> str:
        """
        Add node if absent; merge attributes + provenance if present.
        Uses fuzzy matching to catch near-duplicate names like
        'Swiggy' vs 'Swiggy Limited'.
        """
        # 1. Try fuzzy match against existing labels
        existing_labels = [n["label"] for n in nodes.values()]
        fuzzy_match = fuzzy_match_entity(name, existing_labels, threshold=88.0)
        if fuzzy_match:
            # Resolve to the existing node's ID
            label_idx = _label_index()
            nid = label_idx.get(normalize_name(fuzzy_match))
            if nid and nid in nodes:
                # Merge attributes and provenance into the existing node
                enriched = dict(attributes)
                enriched.update(extract_numeric_attributes(name))
                nodes[nid]["attributes"] = merge_attributes(
                    nodes[nid]["attributes"], enriched
                )
                if chunk_id and chunk_id not in nodes[nid]["provenance"]:
                    nodes[nid]["provenance"] = merge_provenance(
                        nodes[nid]["provenance"], [chunk_id]
                    )
                return nid

        # 2. No fuzzy match – create a new node
        nid = make_node_id(name, node_type)
        # Enrich attributes with any regex-extractable numerics from the name
        enriched = dict(attributes)
        enriched.update(extract_numeric_attributes(name))

        if nid not in nodes:
            nodes[nid] = {
                "id":          nid,
                "type":        node_type,
                "label":       name,
                "attributes":  enriched,
                "type_method": type_method,
                "provenance":  [chunk_id] if chunk_id else [],
            }
        else:
            nodes[nid]["attributes"] = merge_attributes(
                nodes[nid]["attributes"], enriched
            )
            if chunk_id and chunk_id not in nodes[nid]["provenance"]:
                nodes[nid]["provenance"] = merge_provenance(
                    nodes[nid]["provenance"], [chunk_id]
                )
        return nid

    # ── Helper: add an edge ──────────────────────────────────────────────────
    def add_edge(
        source_id: str,
        target_id: str,
        relation: str,
        raw_relation: str,
        chunk_id: str,
        extraction_source: str = "rule",   # "llm" | "rule" | "regex" | "fallback"
    ) -> None:
        """
        Append an edge dict to the edge list.

        confidence is determined by extraction_source:
          llm      → 0.90  (most trusted — model reasoned about it)
          rule     → 0.75  (controlled vocab match)
          regex    → 0.65  (pattern-based extraction)
          fallback → 0.50  (related_to catch-all)

        The final confidence is lowered by 0.10 when the relation is
        the fallback "related_to", regardless of source.
        """
        base_confidence = CONFIDENCE_SCORES.get(extraction_source, 0.75)
        # Penalise fallback relation regardless of source
        confidence = round(base_confidence - 0.10, 2) if relation == "related_to" else base_confidence

        edges.append(
            {
                "source":             source_id,
                "target":             target_id,
                "relation":           relation,
                "raw_relation":       raw_relation,
                "confidence":         confidence,
                "extraction_source":  extraction_source,
                "provenance":         [chunk_id] if chunk_id else [],
            }
        )

    # ── Helper: resolve entity name → existing node id ───────────────────────
    def resolve_name(name: str) -> Optional[str]:
        """
        Try to find an existing node matching the given name.
        Falls back to None if not found.
        """
        norm = normalize_name(name)
        for nid, node in nodes.items():
            if normalize_name(node["label"]) == norm:
                return nid
        return None

    # ── Process each extraction ───────────────────────────────────────────────
    # Track which companies we've seen (to attach events)
    company_ids: List[str] = []
    # Track event nodes per year to merge multi-chunk events
    event_nodes_by_year: Dict[str, str] = {}  # year → event node id

    # First pass: all entities and source nodes
    for extraction in extractions:
        chunk_id = extraction.get("chunk_id", "unknown")

        # ── Entities ────────────────────────────────────────────────────────
        for ent in extraction.get("entities", []):
            name = ent.get("name", "").strip()
            etype = ent.get("type", "").strip()
            if not name or not etype:
                continue

            nid = upsert_node(
                name=name,
                node_type=etype,
                attributes=ent.get("attributes", {}),
                chunk_id=chunk_id,
                type_method=ent.get("type_method", "llm"),
            )

            if etype == "Company" and nid not in company_ids:
                company_ids.append(nid)

        # ── Sources ─────────────────────────────────────────────────────────
        for src_name in extraction.get("sources", []):
            upsert_node(
                name=src_name,
                node_type="Source",
                attributes={},
                chunk_id=chunk_id,
                type_method="rule:Source",
            )

    # Second pass: relations
    for extraction in extractions:
        chunk_id = extraction.get("chunk_id", "unknown")

        for rel in extraction.get("relations", []):
            src_name = rel.get("source", "").strip()
            tgt_name = rel.get("target", "").strip()
            relation = rel.get("relation", "related_to")
            raw = rel.get("raw_relation", relation)

            if not src_name or not tgt_name:
                continue

            # Resolve or create nodes for dangling references
            src_id = resolve_name(src_name)
            if src_id is None:
                logger.debug(f"[build] Auto-creating node for '{src_name}' (unknown type)")
                src_id = upsert_node(src_name, "Unknown", {}, chunk_id, type_method="fallback")

            tgt_id = resolve_name(tgt_name)
            if tgt_id is None:
                logger.debug(f"[build] Auto-creating node for '{tgt_name}' (unknown type)")
                tgt_id = upsert_node(tgt_name, "Unknown", {}, chunk_id, type_method="fallback")

            controlled = map_relation(relation)
            # Relations from LLM extraction get highest confidence
            add_edge(src_id, tgt_id, controlled, raw, chunk_id, extraction_source="llm")

    # Third pass: events
    for extraction in extractions:
        chunk_id = extraction.get("chunk_id", "unknown")

        for evt in extraction.get("events", []):
            year = str(evt.get("year", "")).strip()
            desc = evt.get("description", "").strip()
            tags = evt.get("tags", [])
            involved = evt.get("entities_involved", [])

            if not year or not desc:
                continue

            # ── Merge or create event node for this year ─────────────────
            if year in event_nodes_by_year:
                evt_id = event_nodes_by_year[year]
                existing = nodes[evt_id]
                # Append description if different
                if desc not in existing["attributes"].get("description", ""):
                    existing["attributes"]["description"] = (
                        existing["attributes"].get("description", "")
                        + "; " + desc
                    ).lstrip("; ")
                # Merge tags
                existing_tags = set(existing["attributes"].get("tags", []))
                existing["attributes"]["tags"] = sorted(
                    existing_tags | set(tags)
                )
                # Merge event_types
                existing_types = set(existing["attributes"].get("event_types", []))
                new_types = _classify_event_types(tags)
                existing["attributes"]["event_types"] = sorted(
                    existing_types | set(new_types)
                )
                if chunk_id not in existing["provenance"]:
                    existing["provenance"] = merge_provenance(
                        existing["provenance"], [chunk_id]
                    )
            else:
                evt_id = make_event_id(year)
                event_types = _classify_event_types(tags)
                nodes[evt_id] = {
                    "id": evt_id,
                    "type": "Event",
                    "label": f"Event {year}",
                    "attributes": {
                        "year": year,
                        "description": desc,
                        "tags": tags,
                        # Structured event_type field (new in v2)
                        "event_types": event_types,
                        # Primary event_type for simple queries
                        "primary_event_type": event_types[0] if event_types else "other",
                    },
                    "provenance": [chunk_id],
                }
                event_nodes_by_year[year] = evt_id

                # Connect each known company → has_event
                for company_id in company_ids:
                    add_edge(company_id, evt_id, "has_event", "has_event", chunk_id, extraction_source="rule")

            # ── Link event → involved entities ────────────────────────────
            for entity_name in involved:
                entity_id = resolve_name(entity_name)
                if entity_id is None:
                    continue

                entity_type = nodes[entity_id]["type"]

                # Pick a meaningful relation based on context
                if "Acquisition" in entity_type:
                    rel = "acquired"
                elif entity_type in ("Product", "Feature", "Service", "Programme"):
                    rel = "launched"
                elif any(t in tags for t in ("Fundraise", "Listing")):
                    rel = "related_to"
                else:
                    rel = "launched"

                add_edge(evt_id, entity_id, rel, rel, chunk_id, extraction_source="rule")

    # ── described_in edges (source nodes → all core entities) ────────────────
    source_ids = [
        nid for nid, n in nodes.items() if n["type"] == "Source"
    ]
    for extraction in extractions:
        chunk_id = extraction.get("chunk_id", "unknown")
        # Attach the sources mentioned in this chunk to all entities from it
        chunk_entity_ids = [
            make_node_id(e["name"], e["type"])
            for e in extraction.get("entities", [])
            if e.get("name") and e.get("type") and e.get("type") != "Source"
        ]
        for src_name in extraction.get("sources", []):
            src_id = make_node_id(src_name, "Source")
            if src_id not in nodes:
                continue
            for eid in chunk_entity_ids:
                if eid in nodes:
                    add_edge(eid, src_id, "described_in", "described_in", chunk_id, extraction_source="rule")

    # ── Final deduplication ───────────────────────────────────────────────────
    node_list = dedup_nodes(list(nodes.values()))
    edge_list = dedup_edges(edges)

    # ── Compute per-type statistics for metadata ──────────────────────────────
    type_counts: Dict[str, int] = {}
    for n in node_list:
        t = n.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    confidences = [e.get("confidence", 0.5) for e in edge_list]
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    high_conf_edges = sum(1 for c in confidences if c >= 0.75)

    graph = {
        "nodes": node_list,
        "edges": edge_list,
        "metadata": {
            "created_at":         utc_now(),
            "chunk_count":        len(extractions),
            "node_count":         len(node_list),
            "edge_count":         len(edge_list),
            "chunk_ids":          [e.get("chunk_id", "?") for e in extractions],
            "node_type_counts":   type_counts,
            "event_count":        type_counts.get("Event", 0),
            "avg_edge_confidence": avg_conf,
            "high_confidence_edges": high_conf_edges,
        },
    }

    logger.info(
        f"[build_graph] Built graph: {len(node_list)} nodes, {len(edge_list)} edges"
    )
    return graph
