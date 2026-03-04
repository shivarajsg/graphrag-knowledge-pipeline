"""
query_engine.py
---------------
Programmatic query helpers for traversing the Graph RAG knowledge graph.

All functions take a graph dict (produced by build_graph / loaded from
graph.json) and return plain Python lists — no graph database required.

Usage
-----
    import json
    from graphrag.query_engine import GraphQueryEngine

    with open("outputs/graph.json") as f:
        graph = json.load(f)

    engine = GraphQueryEngine(graph)

    engine.find_products_launched(year=2020)
    engine.find_services_offered()
    engine.find_acquisitions()
    engine.find_capabilities()
    engine.find_events_by_type("acquisition")
    engine.find_entities_by_type("Product")
    engine.find_sources()
    engine.filter_edges_by_confidence(min_confidence=0.75)
    engine.graph_summary()
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class GraphQueryEngine:
    """
    Lightweight traversal engine over a Graph RAG JSON dict.

    All methods return lists of dicts so results are easy to print,
    iterate, or feed into downstream processing.
    """

    def __init__(self, graph: Dict[str, Any]) -> None:
        self._nodes: Dict[str, Dict] = {n["id"]: n for n in graph.get("nodes", [])}
        self._edges: List[Dict] = graph.get("edges", [])
        self._meta: Dict = graph.get("metadata", {})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _edges_from(self, node_id: str, relation: Optional[str] = None) -> List[Dict]:
        return [
            e for e in self._edges
            if e["source"] == node_id
            and (relation is None or e["relation"] == relation)
        ]

    def _edges_to(self, node_id: str, relation: Optional[str] = None) -> List[Dict]:
        return [
            e for e in self._edges
            if e["target"] == node_id
            and (relation is None or e["relation"] == relation)
        ]

    def _resolve(self, node_id: str) -> Optional[Dict]:
        return self._nodes.get(node_id)

    def _nodes_of_type(self, node_type: str) -> List[Dict]:
        return [n for n in self._nodes.values() if n.get("type") == node_type]

    # ── Public query API ──────────────────────────────────────────────────────

    def find_products_launched(self, year: Optional[int | str] = None) -> List[Dict]:
        """
        Return products (and features) launched in a given year.
        If year is None, return all launched products across all years.

        Example
        -------
        engine.find_products_launched(2020)
        → [{"name": "Swiggy Instamart", "type": "Product", "year": "2020", ...}, ...]
        """
        results = []
        event_nodes = self._nodes_of_type("Event")

        for evt in event_nodes:
            evt_year = evt.get("attributes", {}).get("year", "")
            if year is not None and str(evt_year) != str(year):
                continue

            launched_edges = self._edges_from(evt["id"], relation="launched")
            for edge in launched_edges:
                target = self._resolve(edge["target"])
                if target and target.get("type") in ("Product", "Feature", "Service"):
                    results.append({
                        "name":       target["label"],
                        "type":       target["type"],
                        "year":       evt_year,
                        "attributes": target.get("attributes", {}),
                        "confidence": edge.get("confidence", 0.0),
                    })

        return sorted(results, key=lambda x: (x["year"], x["name"]))

    def find_services_offered(self) -> List[Dict]:
        """
        Return all services offered by the platform.

        Traversal: Company →[operates]→ Platform →[offers]→ Service

        Example
        -------
        engine.find_services_offered()
        → [{"name": "Food Delivery", "type": "Service", ...}, ...]
        """
        results = []
        company_nodes = self._nodes_of_type("Company")

        for company in company_nodes:
            for op_edge in self._edges_from(company["id"], "operates"):
                platform = self._resolve(op_edge["target"])
                if not platform:
                    continue
                for offer_edge in self._edges_from(platform["id"], "offers"):
                    service = self._resolve(offer_edge["target"])
                    if service:
                        results.append({
                            "name":       service["label"],
                            "type":       service["type"],
                            "via":        platform["label"],
                            "attributes": service.get("attributes", {}),
                        })

        return results

    def find_acquisitions(self) -> List[Dict]:
        """
        Return all acquired companies/assets.

        Traversal: Company →[acquired]→ Acquisition

        Example
        -------
        engine.find_acquisitions()
        → [{"name": "Dineout", "acquired_year": "2022", ...}, ...]
        """
        results = []
        for edge in self._edges:
            if edge["relation"] != "acquired":
                continue
            acquirer = self._resolve(edge["source"])
            target   = self._resolve(edge["target"])
            if acquirer and target and acquirer.get("type") == "Company":
                attrs = target.get("attributes", {})
                results.append({
                    "name":          target["label"],
                    "acquired_by":   acquirer["label"],
                    "acquired_year": attrs.get("acquired_year", "unknown"),
                    "stake":         attrs.get("stake", "unknown"),
                    "attributes":    attrs,
                    "confidence":    edge.get("confidence", 0.0),
                })

        return sorted(results, key=lambda x: x["acquired_year"])

    def find_capabilities(self) -> List[Dict]:
        """
        Return all capabilities and programmes that power the platform.

        Traversal: Platform →[enabled_by]→ Capability/Programme

        Example
        -------
        engine.find_capabilities()
        → [{"name": "Technology", "type": "Capability", ...}, ...]
        """
        results = []
        platform_nodes = self._nodes_of_type("Platform")

        for platform in platform_nodes:
            for edge in self._edges_from(platform["id"], "enabled_by"):
                cap = self._resolve(edge["target"])
                if cap:
                    results.append({
                        "name":       cap["label"],
                        "type":       cap["type"],
                        "platform":   platform["label"],
                        "attributes": cap.get("attributes", {}),
                    })

        return results

    def find_events_by_type(self, event_type: Optional[str] = None) -> List[Dict]:
        """
        Return timeline events, optionally filtered by event_type.

        event_type values: "launch", "acquisition", "milestone",
                           "fundraise", "listing", "corporate"

        Example
        -------
        engine.find_events_by_type("acquisition")
        → [{"year": "2022", "description": "Acquired Dineout...", ...}]
        """
        events = self._nodes_of_type("Event")
        results = []
        for evt in events:
            attrs = evt.get("attributes", {})
            types = attrs.get("event_types", [])
            if event_type is None or event_type in types:
                results.append({
                    "year":               attrs.get("year", ""),
                    "primary_event_type": attrs.get("primary_event_type", ""),
                    "event_types":        types,
                    "description":        attrs.get("description", ""),
                    "tags":               attrs.get("tags", []),
                })
        return sorted(results, key=lambda x: x["year"])

    def find_entities_by_type(self, entity_type: str) -> List[Dict]:
        """
        Return all nodes of a given type.

        Example
        -------
        engine.find_entities_by_type("Partner")
        → [{"name": "Restaurant Partners", ...}, ...]
        """
        return [
            {
                "name":       n["label"],
                "type":       n["type"],
                "id":         n["id"],
                "attributes": n.get("attributes", {}),
                "provenance": n.get("provenance", []),
            }
            for n in self._nodes_of_type(entity_type)
        ]

    def find_sources(self) -> List[Dict]:
        """
        Return all source documents and the entities they describe.

        Example
        -------
        engine.find_sources()
        → [{"source": "Swiggy Limited IPO Prospectus", "entities": [...]}]
        """
        sources = self._nodes_of_type("Source")
        results = []
        for src in sources:
            described = [
                self._resolve(e["source"])["label"]
                for e in self._edges_to(src["id"], "described_in")
                if self._resolve(e["source"])
            ]
            results.append({
                "source":   src["label"],
                "id":       src["id"],
                "entities": sorted(set(described)),
                "count":    len(set(described)),
            })
        return results

    def filter_edges_by_confidence(self, min_confidence: float = 0.75) -> List[Dict]:
        """
        Return edges at or above a confidence threshold.

        Example
        -------
        engine.filter_edges_by_confidence(0.9)
        → edges extracted by LLM only
        """
        results = []
        for edge in self._edges:
            if edge.get("confidence", 0.0) >= min_confidence:
                src = self._resolve(edge["source"])
                tgt = self._resolve(edge["target"])
                results.append({
                    "from":              src["label"] if src else edge["source"],
                    "relation":          edge["relation"],
                    "to":                tgt["label"] if tgt else edge["target"],
                    "confidence":        edge["confidence"],
                    "extraction_source": edge.get("extraction_source", "unknown"),
                })
        return sorted(results, key=lambda x: x["confidence"], reverse=True)

    def graph_summary(self) -> Dict[str, Any]:
        """
        Return a structured summary of the graph.

        Example
        -------
        engine.graph_summary()
        → {"nodes": 39, "edges": 98, "by_type": {"Product": 9, ...}, ...}
        """
        type_counts: Dict[str, int] = {}
        for n in self._nodes.values():
            t = n.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        confidences = [e.get("confidence", 0.5) for e in self._edges]
        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        relation_counts: Dict[str, int] = {}
        for e in self._edges:
            r = e.get("relation", "unknown")
            relation_counts[r] = relation_counts.get(r, 0) + 1

        return {
            "nodes":              len(self._nodes),
            "edges":              len(self._edges),
            "by_type":            dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            "avg_edge_confidence": avg_conf,
            "relation_counts":    dict(sorted(relation_counts.items(), key=lambda x: -x[1])),
            "chunks_processed":   self._meta.get("chunk_count", 0),
            "created_at":         self._meta.get("created_at", ""),
        }

    def print_summary(self) -> None:
        """Pretty-print a human-readable graph summary to stdout."""
        s = self.graph_summary()
        print("\n" + "─" * 48)
        print("  Graph Summary")
        print("─" * 48)
        print(f"  Nodes            : {s['nodes']}")
        print(f"  Edges            : {s['edges']}")
        print(f"  Avg confidence   : {s['avg_edge_confidence']}")
        print(f"  Chunks processed : {s['chunks_processed']}")
        print()
        print("  Nodes by type:")
        for ntype, count in s["by_type"].items():
            bar = "█" * count
            print(f"    {ntype:<22} {count:>3}  {bar}")
        print()
        print("  Top relations:")
        for rel, count in list(s["relation_counts"].items())[:8]:
            print(f"    {rel:<30} {count:>3}")
        print("─" * 48 + "\n")


# ── Convenience loader ────────────────────────────────────────────────────────

def load_engine(graph_path: str = "outputs/graph.json") -> GraphQueryEngine:
    """
    Load a graph from a JSON file and return a ready-to-use engine.

    Example
    -------
    from graphrag.query_engine import load_engine
    engine = load_engine("outputs/graph.json")
    engine.find_products_launched(2020)
    """
    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)
    return GraphQueryEngine(graph)
