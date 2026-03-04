"""
ontology.py
-----------
Controlled vocabulary for the Graph RAG pipeline.

v3: All definitions are now loaded from ontology_config.json at the
project root, making the ontology fully configurable without touching
Python source code.

Python constants are derived from the JSON at import time and remain
available under the same names as before — so no other module needs
to change its import statements.

To extend the ontology: edit ontology_config.json only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List

logger = logging.getLogger(__name__)

# ── Load ontology_config.json ─────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    """
    Search for ontology_config.json in:
      1. The project root (two levels above this file)
      2. The same directory as this file (fallback for installed packages)
    """
    candidates = [
        Path(__file__).parent.parent / "ontology_config.json",   # project root
        Path(__file__).parent / "ontology_config.json",           # package dir
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            logger.debug(f"[ontology] Loaded config from {path}")
            return cfg

    logger.warning(
        "[ontology] ontology_config.json not found — using built-in defaults. "
        "Place the file at the project root to enable full configurability."
    )
    return _builtin_defaults()


def _builtin_defaults() -> Dict[str, Any]:
    """
    Minimal hard-coded defaults used only when the JSON file is missing.
    These mirror the ontology_config.json exactly, so the pipeline never
    breaks even without the config file present.
    """
    return {
        "entity_types": [
            "Company", "Platform", "Service", "Product", "Feature",
            "Partner", "Capability", "Programme", "Acquisition",
            "Event", "Source", "Person", "Location", "Financial", "Entity",
        ],
        "relations": [
            {"name": r} for r in [
                "operates", "offers", "enabled_by", "supported_by", "includes",
                "integrated_surface", "executed_via", "acquired", "launched",
                "has_event", "expanded_to", "partnered_with", "benefits_across",
                "feature", "described_in", "events_experiences_surface",
                "expanded_surface", "related_to",
            ]
        ],
        "relation_synonyms": {
            "provides": "offers", "delivers": "offers", "supply": "offers",
            "is part of": "includes", "encompasses": "includes",
            "contains": "includes", "comprises": "includes",
            "integrated with": "integrated_surface",
            "integrated into": "integrated_surface",
            "through": "executed_via", "via": "executed_via",
            "uses": "enabled_by", "powered by": "enabled_by",
            "underpinned by": "enabled_by", "built on": "enabled_by",
            "supports": "supported_by", "backed by": "supported_by",
            "bought": "acquired", "purchased": "acquired",
            "merged with": "acquired", "took stake in": "acquired",
            "introduced": "launched", "released": "launched",
            "debuted": "launched", "rolled out": "launched",
            "expanded to": "expanded_to", "scaled to": "expanded_to",
            "mentioned in": "described_in", "sourced from": "described_in",
            "benefits": "benefits_across",
            "has feature": "feature", "sub-feature": "feature",
            "related to": "related_to", "associated with": "related_to",
        },
        "confidence_scores": {
            "llm": 0.90, "rule": 0.75, "regex": 0.65, "fallback": 0.50
        },
        "event_tags": [
            "Corporate", "Launch", "Pioneer", "Acquisition",
            "Milestone", "Scale", "Fundraise", "Listing",
        ],
        "mermaid_sections": [
            {"label": "Core Entities",             "types": ["Company", "Platform"]},
            {"label": "Services",                  "types": ["Service"]},
            {"label": "Products & Features",       "types": ["Product", "Feature"]},
            {"label": "Partners",                  "types": ["Partner"]},
            {"label": "Capabilities & Programmes", "types": ["Capability", "Programme"]},
            {"label": "Acquisitions",              "types": ["Acquisition"]},
            {"label": "Events / Timeline",         "types": ["Event"]},
            {"label": "Sources",                   "types": ["Source"]},
            {"label": "Other",                     "types": ["Person", "Location", "Financial", "Entity"]},
        ],
    }


# ── Build module-level constants from config ──────────────────────────────────

_CFG: Dict[str, Any] = _load_config()

# Frozenset of valid relation names (used by map_relation())
CONTROLLED_RELATIONS: FrozenSet[str] = frozenset(
    r["name"] if isinstance(r, dict) else r
    for r in _CFG.get("relations", [])
)

# Synonym map: free-text phrase → controlled relation name
RELATION_SYNONYM_MAP: Dict[str, str] = dict(_CFG.get("relation_synonyms", {}))

# Ordered list of node type strings
NODE_TYPES: List[str] = list(_CFG.get("entity_types", []))

# Mermaid section ordering: list of (section_label, [types]) tuples
MERMAID_SECTION_ORDER: List = [
    (s["label"], s["types"])
    for s in _CFG.get("mermaid_sections", [])
]

# Allowed event tag strings
EVENT_TAGS: List[str] = list(_CFG.get("event_tags", []))

# Confidence score table (used by graph_builder for edge scoring)
CONFIDENCE_SCORES: Dict[str, float] = dict(
    _CFG.get("confidence_scores", {"llm": 0.9, "rule": 0.75, "regex": 0.65, "fallback": 0.5})
)

# ── LLM ontology description (injected into extraction prompt) ────────────────
# Built dynamically from the loaded config so it stays in sync automatically.

def _build_ontology_description() -> str:
    lines = ["NODE TYPES:"]
    type_descriptions = {
        "Company":     "The primary organization being analysed",
        "Platform":    "A unified digital/technology platform operated by the company",
        "Service":     "A broad service category offered to end users",
        "Product":     "A specific branded product or sub-service",
        "Feature":     "A specific feature within a product/service",
        "Partner":     "External parties (restaurants, merchants, delivery agents, brands)",
        "Capability":  "Core organisational/technical capabilities",
        "Programme":   "Membership, loyalty, or subscription programs",
        "Acquisition": "An acquired company or asset",
        "Event":       "A dated occurrence (launch, acquisition, funding, milestone)",
        "Source":      "The document from which the information is sourced",
        "Person":      "Named individuals (founders, executives)",
        "Location":    "Geographic entities",
        "Financial":   "Financial metrics, funding rounds, valuations",
        "Entity":      "Generic fallback type when no other type fits",
    }
    for t in NODE_TYPES:
        desc = type_descriptions.get(t, t)
        lines.append(f"- {t:<14}: {desc}")

    lines.append("\nCONTROLLED RELATIONS (use EXACTLY one of these strings):")
    for r in _CFG.get("relations", []):
        name = r["name"] if isinstance(r, dict) else r
        desc = r.get("description", "") if isinstance(r, dict) else ""
        example = r.get("example", "") if isinstance(r, dict) else ""
        line = f"- {name:<30}"
        if desc:
            line += f" : {desc}"
        if example:
            line += f"  (e.g. {example})"
        lines.append(line)

    lines.append(f"\nEVENT TAGS (use from this list only):\n{', '.join(EVENT_TAGS)}")
    return "\n".join(lines)


ONTOLOGY_DESCRIPTION: str = _build_ontology_description()
