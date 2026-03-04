"""
classifier.py
-------------
Rule-based entity type classification layer.

Why this exists
---------------
LLM extraction assigns types, but LLMs can be inconsistent — especially
for short entity names extracted without full context (e.g. "Instamart"
alone gives no type signal). This module provides a deterministic
rule-based fallback that runs *after* LLM/mock extraction and corrects
or fills in missing types.

Public API
----------
    classify_entity_type(name, context_text)  → str
    classify_entities(entities, chunk_text)   → list[dict]  (in-place type repair)

Classification strategy
-----------------------
Rules fire in priority order. The first matching rule wins.
Each rule checks:
  1. The entity name itself (e.g. "prospectus" → Source)
  2. A ±80-character window around the entity name in the chunk text
     (so "Instamart" near "launched" → Product)

If no rule matches, the original LLM-assigned type is kept.
If the original type is also missing/unknown, fallback is "Entity".
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Classification rule table ─────────────────────────────────────────────────
#
# Each rule is a tuple:
#   (target_type, name_keywords, context_keywords)
#
# name_keywords    : list of substrings checked against the entity name (lower)
# context_keywords : list of substrings checked against the ±80-char context window
#
# A rule fires when ANY name_keyword matches OR ANY context_keyword matches.
# Rules are evaluated top-to-bottom; first match wins.

_RULES: List[Tuple[str, List[str], List[str]]] = [
    # ── Source documents ──────────────────────────────────────────────────────
    (
        "Source",
        ["prospectus", "annual report", "ipo", "filing", "10-k", "10-q",
         "press release", "whitepaper"],
        ["source of this information", "published by", "filed with", "the report"],
    ),

    # ── Company ───────────────────────────────────────────────────────────────
    (
        "Company",
        ["limited", " ltd", " inc", " corp", " plc", " llp"],
        ["incorporated", "company was founded", "the company is",
         "parent company", "holding company"],
    ),

    # ── Platform ─────────────────────────────────────────────────────────────
    (
        "Platform",
        ["platform", "ecosystem", "marketplace"],
        ["unified platform", "operates a platform", "built a platform",
         "convenience platform", "digital platform"],
    ),

    # ── Programme / Membership ────────────────────────────────────────────────
    (
        "Programme",
        ["one", "pass", "prime", "membership", "loyalty", "subscription", "plus"],
        ["membership programme", "subscription plan", "loyalty program",
         "membership base", "programme providing benefits"],
    ),

    # ── Acquisition ───────────────────────────────────────────────────────────
    (
        "Acquisition",
        [],   # rely on context alone
        ["acquired", "acquisition of", "purchased", "bought", "stake in",
         "merged with", "100% stake", "takeover"],
    ),

    # ── Partner ───────────────────────────────────────────────────────────────
    (
        "Partner",
        ["partner", "partners", "merchant", "vendor", "supplier", "restaurant"],
        ["restaurant partners", "merchant partners", "brand partners",
         "delivery partners", "partnered with", "partner network"],
    ),

    # ── Capability ────────────────────────────────────────────────────────────
    (
        "Capability",
        ["analytics", "fulfilment", "fulfillment", "technology", "tech stack",
         "machine learning", "ai", "data"],
        ["foundational capabilit", "core capabilit", "underpinning",
         "tech stack", "reusable tech", "powered by"],
    ),

    # ── Feature (before Product, as features are more specific) ───────────────
    (
        "Feature",
        ["bolt", "10-min", "10 min", "quick bites", "quick-bites"],
        ["10-minute delivery", "feature within", "sub-feature", "powered feature"],
    ),

    # ── Product ───────────────────────────────────────────────────────────────
    (
        "Product",
        ["instamart", "genie", "minis", "mall", "scenes", "snacc", "pyng",
         "dineout", "credit card", "co-branded"],
        ["launched", "introduced", "rolled out", "available in",
         "cities", "dark store", "quick commerce", "scaled to"],
    ),

    # ── Service (broad categories) ────────────────────────────────────────────
    (
        "Service",
        ["delivery", "dining", "grocery", "hyperlocal", "reservations",
         "payments", "events", "experiences"],
        ["service category", "primary service", "offers", "provides service",
         "food delivery", "dining out"],
    ),
]


# ── Context window extraction ─────────────────────────────────────────────────

def _context_window(entity_name: str, text: str, window: int = 80) -> str:
    """
    Return a ±window-character substring of `text` centred on the first
    occurrence of `entity_name`.  Falls back to the full text if not found.
    """
    if not text:
        return ""
    idx = text.lower().find(entity_name.lower())
    if idx == -1:
        return text  # name not in text — use full chunk
    start = max(0, idx - window)
    end = min(len(text), idx + len(entity_name) + window)
    return text[start:end]


# ── Core classification logic ─────────────────────────────────────────────────

def classify_entity_type(
    entity_name: str,
    context_text: str = "",
    existing_type: str = "",
) -> Tuple[str, str]:
    """
    Infer the entity type for a named entity.

    Parameters
    ----------
    entity_name   : the raw entity name (e.g. "Swiggy Instamart")
    context_text  : the full chunk text in which this entity appears
    existing_type : the type already assigned (by LLM or mock data);
                    kept if rules don't override it

    Returns
    -------
    (final_type, method)
    Where method is one of:
        "llm"          — LLM type kept (no rule overrode it)
        "rule:<type>"  — rule-based classification applied
        "fallback"     — no LLM type and no rule matched

    Examples
    --------
    classify_entity_type("Swiggy Instamart",
                         "In 2020, Instamart was launched in 2 cities")
    → ("Product", "rule:Product")

    classify_entity_type("Swiggy", "Swiggy Limited is incorporated in …",
                         existing_type="Company")
    → ("Company", "llm")   # LLM already correct, no need to override
    """
    name_lower = entity_name.lower()
    ctx_lower = _context_window(entity_name, context_text).lower()

    for target_type, name_kws, ctx_kws in _RULES:
        # Skip rules that would demote a perfectly good known type
        # (we only override "Unknown", "", or a type that clearly misfits)
        name_hit = any(kw in name_lower for kw in name_kws)
        ctx_hit  = any(kw in ctx_lower  for kw in ctx_kws)

        if name_hit or ctx_hit:
            if existing_type in ("", "Unknown", "Entity"):
                # LLM gave us nothing useful — take the rule's type
                logger.debug(
                    f"[classifier] '{entity_name}': no LLM type → "
                    f"rule assigned '{target_type}'"
                )
                return target_type, f"rule:{target_type}"

            if existing_type == target_type:
                # Rule agrees with LLM — keep LLM label (higher confidence)
                return existing_type, "llm"

            # Rule disagrees — only override if existing_type is generic
            if existing_type in ("Entity", "Unknown", "Other"):
                logger.debug(
                    f"[classifier] '{entity_name}': LLM type='{existing_type}' "
                    f"overridden by rule → '{target_type}'"
                )
                return target_type, f"rule:{target_type}"

            # LLM type looks specific — trust it
            return existing_type, "llm"

    # No rule matched
    if existing_type and existing_type not in ("", "Unknown", "Entity"):
        return existing_type, "llm"

    logger.debug(f"[classifier] '{entity_name}': no rule matched, falling back to 'Entity'")
    return "Entity", "fallback"


# ── Batch helper ──────────────────────────────────────────────────────────────

def classify_entities(
    entities: List[Dict[str, Any]],
    chunk_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Apply classify_entity_type() to a list of entity dicts in-place.

    Adds a "type_method" attribute to each entity recording how the type
    was determined (useful for debugging and confidence scoring).

    Parameters
    ----------
    entities   : list of entity dicts with at least 'name' and 'type'
    chunk_text : the source chunk text (used for context window lookup)

    Returns
    -------
    The same list with 'type' and 'type_method' fields updated.
    """
    for ent in entities:
        name          = ent.get("name", "")
        existing_type = ent.get("type", "")

        final_type, method = classify_entity_type(
            entity_name=name,
            context_text=chunk_text,
            existing_type=existing_type,
        )

        ent["type"]        = final_type
        ent["type_method"] = method

    return entities
