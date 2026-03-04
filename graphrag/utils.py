"""
utils.py
--------
Shared helper functions used across the pipeline.

Improvements (v2)
-----------------
- Fuzzy entity normalization via rapidfuzz (falls back to difflib SequenceMatcher)
- Stable node IDs: slug + MD5 short-hash to survive name variations
- Regex-based numeric attribute extraction (cities, users, members, SKUs, etc.)
"""

import re
import json
import hashlib
import difflib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .ontology import CONTROLLED_RELATIONS, RELATION_SYNONYM_MAP


# ── Fuzzy similarity backend (rapidfuzz preferred, difflib fallback) ───────────

def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Return a 0–100 similarity score between two strings.
    Uses rapidfuzz if installed, otherwise difflib.SequenceMatcher.
    """
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b)
    except ImportError:
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        return ratio * 100.0


# ── ID / naming helpers ───────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Lowercase and strip common corporate suffixes so that
    'Swiggy' and 'Swiggy Limited' resolve to the same base name.
    """
    if not name:
        return ""
    n = name.strip()
    for suffix in [
        " limited", " ltd.", " ltd", " inc.", " inc",
        " corp.", " corp", " co.", " plc", " llp", " llc",
    ]:
        if n.lower().endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n.lower().strip()


def fuzzy_match_entity(
    name: str,
    existing_labels: List[str],
    threshold: float = 88.0,
) -> Optional[str]:
    """
    Return the best-matching label from existing_labels if similarity
    exceeds threshold (0–100), else None.

    This prevents duplicates like "Swiggy" / "Swiggy Limited" / "Swiggy Ltd"
    from entering the graph as separate nodes.

    Uses rapidfuzz if available, falls back to difflib.

    Examples
    --------
    fuzzy_match_entity("Swiggy Ltd", ["Swiggy", "Instamart"])
    → "Swiggy"   (ratio ≈ 91)

    fuzzy_match_entity("Zomato", ["Swiggy", "Instamart"])
    → None        (no match above threshold)
    """
    norm_input = normalize_name(name)
    best_label: Optional[str] = None
    best_score: float = 0.0

    for label in existing_labels:
        norm_label = normalize_name(label)
        score = _fuzzy_ratio(norm_input, norm_label)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= threshold:
        return best_label
    return None


def make_node_id(name: str, node_type: str = "") -> str:
    """
    Generate a stable, collision-resistant node ID.

    Format: {type}_{slug}_{6-char hash}

    The hash component ensures that near-identical names with different
    types (e.g. "Technology" as Capability vs. as a generic word) stay
    distinct, while the slug keeps IDs human-readable.

    Examples
    --------
    make_node_id("Swiggy Instamart", "Product") → "product_swiggy_instamart_a3f9c1"
    make_node_id("Swiggy Limited",   "Company") → "company_swiggy_8b2d4e"
    """
    normalized = normalize_name(name)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not slug:
        slug = "node"
    short_hash = hashlib.md5(f"{node_type}:{normalized}".encode()).hexdigest()[:6]
    prefix = (node_type.lower() + "_") if node_type else ""
    return f"{prefix}{slug}_{short_hash}"


def make_event_id(year: str | int, sequence: int = 0) -> str:
    """
    Generate a short event node ID: E2013, E2014_1, …
    Sequence > 0 is used when multiple events share the same year.
    """
    if sequence:
        return f"E{year}_{sequence}"
    return f"E{year}"


# ── Relation mapping ──────────────────────────────────────────────────────────

def map_relation(raw: str) -> str:
    """
    Map a free-text relation to the controlled vocabulary.

    Strategy
    --------
    1. Exact match in controlled set
    2. Exact match in synonym map
    3. Longest partial key match in synonym map
    4. Substring scan of controlled set labels
    5. Fallback → "related_to"
    """
    raw_lower = raw.lower().strip()

    # 1. Direct hit
    if raw_lower in CONTROLLED_RELATIONS:
        return raw_lower

    # 2. Synonym exact
    if raw_lower in RELATION_SYNONYM_MAP:
        return RELATION_SYNONYM_MAP[raw_lower]

    # 3. Synonym partial (longest matching key wins)
    best_key, best_val = "", "related_to"
    for key, val in RELATION_SYNONYM_MAP.items():
        if key in raw_lower and len(key) > len(best_key):
            best_key, best_val = key, val
    if best_key:
        return best_val

    # 4. Controlled set substring
    for rel in CONTROLLED_RELATIONS:
        if rel in raw_lower:
            return rel

    return "related_to"


# ── Rule-based relation extraction from raw text ─────────────────────────────

# Keyword → controlled relation (complements the LLM extraction).
# Applied in extract_relations_from_text() as a lightweight backup.
RULE_BASED_RELATION_PATTERNS: Dict[str, List[str]] = {
    "offers":      ["offers", "provides", "delivers", "services"],
    "acquired":    ["acquired", "purchased", "bought", "merged with", "took stake in"],
    "launched":    ["launched", "introduced", "released", "debuted", "rolled out"],
    "operates":    ["operates", "runs", "manages"],
    "enabled_by":  ["powered by", "underpinned by", "built on", "uses"],
    "includes":    ["includes", "encompasses", "comprises", "contains"],
    "expanded_to": ["expanded to", "scaled to", "grew to"],
    "partnered_with": ["partnered with", "in partnership with"],
}


def extract_relations_from_text(text: str) -> List[str]:
    """
    Scan raw text for keyword-based relation signals and return a list
    of matched controlled relation strings.  Used as a rule-based backup
    when the LLM relation mapping is uncertain.
    """
    found = []
    lower = text.lower()
    for relation, keywords in RULE_BASED_RELATION_PATTERNS.items():
        for kw in keywords:
            if kw in lower:
                found.append(relation)
                break
    return found


# ── Regex numeric attribute extraction ───────────────────────────────────────

# Each entry: (attribute_key, compiled_regex, value_group_index)
_NUMERIC_PATTERNS: List[Tuple[str, Any, int]] = [
    ("cities",          re.compile(r"(\d[\d,]*)\s*cit(?:y|ies)", re.I),          1),
    ("dark_stores",     re.compile(r"(\d[\d,]*)\s*dark\s+stores?", re.I),         1),
    ("skus",            re.compile(r"(\d[\d,]*)\+?\s*SKUs?", re.I),               1),
    ("users_million",   re.compile(r"(\d[\d,]*)\s*million\s+(?:transacted\s+)?users?", re.I), 1),
    ("members_million", re.compile(r"(\d[\d,.]*)\s*million\s+members?", re.I),    1),
    ("members",         re.compile(r"(\d[\d,.]*)\s*[Mm](?:illion)?\s*members?"),  1),
    ("launch_year",     re.compile(r"\b(20\d{2})\b"),                             1),
    ("cities_count",    re.compile(r"(\d+)\s+cities", re.I),                      1),
    ("fundraise_year",  re.compile(r"(20\d{2}|19\d{2}).*?fundrais", re.I),        1),
    ("stake_percent",   re.compile(r"(\d+)%\s+stake", re.I),                      1),
]


def extract_numeric_attributes(text: str) -> Dict[str, str]:
    """
    Extract structured numeric/factual attributes from a raw text snippet.

    Examples
    --------
    extract_numeric_attributes("Instamart scaled to 124 cities")
    → {"cities": "124", "cities_count": "124"}

    extract_numeric_attributes("5.7 million members, public listing in 2024")
    → {"members_million": "5.7", "launch_year": "2024"}
    """
    attrs: Dict[str, str] = {}
    for key, pattern, group in _NUMERIC_PATTERNS:
        m = pattern.search(text)
        if m:
            val = m.group(group).replace(",", "")
            attrs[key] = val
    return attrs


# ── Attribute / provenance merging ────────────────────────────────────────────

def merge_attributes(base: Dict, update: Dict) -> Dict:
    """Merge two attribute dicts; existing non-empty values are preserved."""
    merged = dict(base)
    for k, v in update.items():
        if k not in merged or merged[k] in (None, "", [], {}):
            merged[k] = v
    return merged


def merge_provenance(a: List[str], b: List[str]) -> List[str]:
    """Deduplicated, sorted union of two provenance lists."""
    return sorted(set(a) | set(b))


# ── JSON / LLM output parsing ─────────────────────────────────────────────────

def safe_parse_json(text: str) -> Optional[Dict]:
    """
    Robustly extract a JSON object from an LLM response that may include
    prose, markdown fences, or trailing commentary.
    """
    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: strip markdown fences
    for pattern in [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

    # Attempt 3: find outermost {...}
    m = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── Graph JSON helpers ────────────────────────────────────────────────────────

def dedup_nodes(nodes: List[Dict]) -> List[Dict]:
    """
    Deduplicate node list by id, merging attributes and provenance.
    Later entries win on attribute values when the base value is empty.
    """
    seen: Dict[str, Dict] = {}
    for node in nodes:
        nid = node["id"]
        if nid not in seen:
            seen[nid] = dict(node)
        else:
            seen[nid]["attributes"] = merge_attributes(
                seen[nid].get("attributes", {}),
                node.get("attributes", {}),
            )
            seen[nid]["provenance"] = merge_provenance(
                seen[nid].get("provenance", []),
                node.get("provenance", []),
            )
    return list(seen.values())


def dedup_edges(edges: List[Dict]) -> List[Dict]:
    """
    Deduplicate edge list by (source, target, relation) triple.
    When duplicates are found:
      - provenance lists are merged
      - confidence takes the MAX (most trusted source wins)
      - extraction_source from the highest-confidence entry is kept
    """
    seen: Dict[tuple, Dict] = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["relation"])
        if key not in seen:
            seen[key] = dict(edge)
        else:
            seen[key]["provenance"] = merge_provenance(
                seen[key].get("provenance", []),
                edge.get("provenance", []),
            )
            # Keep the higher confidence value and its source label
            existing_conf = seen[key].get("confidence", 0.5)
            new_conf      = edge.get("confidence", 0.5)
            if new_conf > existing_conf:
                seen[key]["confidence"]        = new_conf
                seen[key]["extraction_source"] = edge.get("extraction_source", "rule")
    return list(seen.values())


# ── Misc ──────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_mermaid_id(node_id: str) -> str:
    """
    Ensure a node ID is a valid Mermaid identifier.
    Mermaid accepts [A-Za-z0-9_-] but some renderers choke on hyphens.
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", node_id)


def escape_mermaid_label(text: str) -> str:
    """
    Escape characters that would break Mermaid label strings.
    HTML inside double-quoted labels is generally fine; we only
    need to guard against unmatched quotes.
    """
    return text.replace('"', "&quot;")
