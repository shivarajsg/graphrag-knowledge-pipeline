"""
extractor.py
------------
Implements extract() – the first stage of the pipeline.

Given a single text chunk (dict with at least 'id' and 'text' keys),
extract() returns a structured dict containing:
    - entities   : list of {name, type, attributes}
    - relations  : list of {source, target, relation}
    - events     : list of {year, description, tags, entities_involved}
    - sources    : list of source document names

Two modes
---------
  mock=True  → Returns pre-extracted data (no API call, great for testing)
  mock=False → Calls Anthropic Claude via the API (requires api_key)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

DEFAULT_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

from .ontology import ONTOLOGY_DESCRIPTION, CONTROLLED_RELATIONS, RELATION_SYNONYM_MAP
from .utils import safe_parse_json, map_relation
from .mock_data import MOCK_EXTRACTIONS
from .classifier import classify_entities

logger = logging.getLogger(__name__)

# ── LLM prompt templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a knowledge-graph extraction engine specialising in corporate documents.

Given a text chunk, extract structured information for building a Graph RAG knowledge graph.

{ONTOLOGY_DESCRIPTION}

RULES:
1. Extract ALL named entities, including implicit ones implied by context.
2. Use ONLY the controlled relation strings listed above. Map any free-text relation.
3. For events, always capture: year (YYYY), short description, tags (from the allowed list),
   and which entity names are directly involved.
4. When a source document is cited, include it in "sources".
5. Do NOT hallucinate entities or relations not present in the text.
6. Attributes should be concise key-value pairs (strings only).

OUTPUT: respond with a single JSON object and NOTHING ELSE – no prose, no markdown fences.

Schema:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "relation": "controlled_relation"}}
  ],
  "events": [
    {{"year": "YYYY", "description": "...", "tags": ["Tag1"], "entities_involved": ["entity_name"]}}
  ],
  "sources": ["source document name"]
}}
"""

USER_PROMPT_TEMPLATE = """Extract knowledge-graph data from the following chunk.
Chunk ID: {chunk_id}

TEXT:
{text}
"""


# ── Public API ────────────────────────────────────────────────────────────────

def extract(
    chunk: Dict[str, Any],
    *,
    mock: bool = False,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract entities, relations, and events from a single chunk.

    Parameters
    ----------
    chunk   : dict with at least 'id' and 'text'
    mock    : if True, return pre-baked data (no API call)
    api_key : Anthropic API key (required when mock=False)
    model   : Anthropic model to use (defaults to ANTHROPIC_MODEL env var or DEFAULT_MODEL)

    Returns
    -------
    dict with keys: chunk_id, entities, relations, events, sources
    """
    chunk_id = str(chunk.get("id", "unknown"))

    if mock:
        return _mock_extract(chunk_id)

    if api_key is None:
        raise ValueError(
            "api_key is required when mock=False. "
            "Pass --mock to run without an API key."
        )

    selected_model = model or DEFAULT_MODEL
    raw = _llm_extract(chunk, api_key=api_key, model=selected_model)
    return _normalise_extraction(raw, chunk_id)



# ── Mock mode ─────────────────────────────────────────────────────────────────

def _mock_extract(chunk_id: str) -> Dict[str, Any]:
    """Return the pre-extracted mock data for the given chunk ID."""
    if chunk_id in MOCK_EXTRACTIONS:
        logger.info(f"[mock] Returning pre-extracted data for chunk {chunk_id}")
        result = dict(MOCK_EXTRACTIONS[chunk_id])
        # Apply classifier even in mock mode — validates and enriches types
        result["entities"] = classify_entities(result.get("entities", []), chunk_text="")
        return result

    logger.warning(
        f"[mock] No pre-extracted data for chunk_id={chunk_id!r}. "
        "Returning empty extraction."
    )
    return {
        "chunk_id": chunk_id,
        "entities": [],
        "relations": [],
        "events": [],
        "sources": [],
    }


# ── LLM mode ─────────────────────────────────────────────────────────────────

def _llm_extract(
    chunk: Dict[str, Any],
    *,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    """Call the Anthropic API to extract structured data from a chunk."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic package is required for LLM mode. "
            "Install with: pip install anthropic"
        ) from e

    chunk_id = str(chunk.get("id", "unknown"))
    text = chunk.get("text", "")

    client = anthropic.Anthropic(api_key=api_key)

    logger.info(f"[llm] Calling {model} for chunk {chunk_id} …")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    chunk_id=chunk_id,
                    text=text,
                ),
            }
        ],
    )

    raw_text = response.content[0].text
    parsed = safe_parse_json(raw_text)

    if parsed is None:
        logger.error(
            f"[llm] Failed to parse JSON from LLM response for chunk {chunk_id}. "
            f"Raw response snippet: {raw_text[:300]}"
        )
        return {
            "chunk_id": chunk_id,
            "entities": [],
            "relations": [],
            "events": [],
            "sources": [],
        }

    parsed["chunk_id"] = chunk_id
    parsed["_chunk_text"] = text   # passed to classifier in _normalise_extraction
    return parsed


# ── Normalisation (post-LLM cleaning) ────────────────────────────────────────

def _normalise_extraction(raw: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """
    Clean and validate an LLM extraction dict:
    - Ensure all required keys are present
    - Map relations to controlled vocabulary
    - Filter out malformed entries with a warning
    """
    out: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "entities": [],
        "relations": [],
        "events": [],
        "sources": [],
    }

    # ── Entities ────────────────────────────────────────────────────────────
    for ent in raw.get("entities", []):
        name = ent.get("name", "").strip()
        etype = ent.get("type", "").strip()
        if not name or not etype:
            logger.debug(f"[normalise] Skipping entity with missing name/type: {ent}")
            continue
        out["entities"].append(
            {
                "name": name,
                "type": etype,
                "attributes": dict(ent.get("attributes", {})),
            }
        )

    # ── Apply rule-based type classification as a post-LLM layer ────────────
    # Passes the full chunk text so classify_entities() can use context windows.
    chunk_text = raw.get("_chunk_text", "")
    out["entities"] = classify_entities(out["entities"], chunk_text)

    # ── Relations ───────────────────────────────────────────────────────────
    for rel in raw.get("relations", []):
        source = rel.get("source", "").strip()
        target = rel.get("target", "").strip()
        relation = rel.get("relation", "").strip()
        if not source or not target:
            logger.debug(f"[normalise] Skipping relation with missing source/target: {rel}")
            continue
        controlled = map_relation(relation)
        out["relations"].append(
            {
                "source": source,
                "target": target,
                "relation": controlled,
                "raw_relation": relation,
            }
        )

    # ── Events ──────────────────────────────────────────────────────────────
    for evt in raw.get("events", []):
        year = str(evt.get("year", "")).strip()
        desc = evt.get("description", "").strip()
        if not year or not desc:
            logger.debug(f"[normalise] Skipping event with missing year/description: {evt}")
            continue
        out["events"].append(
            {
                "year": year,
                "description": desc,
                "tags": list(evt.get("tags", [])),
                "entities_involved": list(evt.get("entities_involved", [])),
            }
        )

    # ── Sources ─────────────────────────────────────────────────────────────
    out["sources"] = [s.strip() for s in raw.get("sources", []) if s.strip()]

    return out
