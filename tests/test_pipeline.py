"""
tests/test_pipeline.py
----------------------
End-to-end and unit tests for the Graph RAG pipeline.
All tests run in mock mode (no API key required).

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_pipeline.py
"""

import json
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


from graphrag.extractor     import extract
from graphrag.graph_builder  import build_graph
from graphrag.renderer       import render_mermaid
from graphrag.retriever      import retrieve_chunks, get_active_backend
from graphrag.pipeline       import run_pipeline
from graphrag.classifier     import classify_entity_type, classify_entities
from graphrag.utils          import (
    make_node_id, normalize_name, map_relation,
    fuzzy_match_entity, extract_numeric_attributes,
    extract_relations_from_text,
)
from graphrag.ontology       import (
    CONTROLLED_RELATIONS, RELATION_SYNONYM_MAP,
    NODE_TYPES, CONFIDENCE_SCORES, MERMAID_SECTION_ORDER,
)


# ── Test chunks ───────────────────────────────────────────────────────────────

CHUNKS = [
    {
        "id": "5",
        "page": 1,
        "text": (
            "Swiggy is a new-age, consumer-first technology company offering users an "
            "easy-to-use unified platform, structured as a consumer convenience platform."
        ),
    },
    {
        "id": "6",
        "page": 2,
        "text": (
            "Swiggy positions itself as a pioneer in high-frequency hyperlocal commerce "
            "categories. Swiggy Instamart scaled to 124 cities as of 31 March 2025."
        ),
    },
    {
        "id": "7",
        "page": 3,
        "text": (
            "The urban convenience platform was built beginning with incorporation in 2013. "
            "In 2020, Swiggy Instamart and Swiggy Genie were launched. "
            "The source of this information is the Swiggy Limited IPO Prospectus."
        ),
    },
]


# ── Utility tests ─────────────────────────────────────────────────────────────

def test_normalize_name():
    assert normalize_name("Swiggy Limited") == "swiggy"
    assert normalize_name("Acme Corp.") == "acme"
    assert normalize_name("  Foo Inc  ") == "foo"
    print("  ✓ normalize_name")


def test_make_node_id():
    nid = make_node_id("Swiggy Instamart", "Product")
    # v2 format: {type}_{slug}_{6-char-hash}
    assert nid.startswith("product_swiggy_instamart_"), f"Unexpected ID: {nid}"
    assert len(nid.split("_")[-1]) == 6, "Hash suffix should be 6 chars"
    # Company name normalisation strips 'Limited'
    nid2 = make_node_id("Swiggy Limited", "Company")
    assert nid2.startswith("company_swiggy_"), f"Unexpected ID: {nid2}"
    # Stability: same inputs → same ID
    assert make_node_id("Swiggy Instamart", "Product") == nid
    print(f"  ✓ make_node_id: {nid}")  


def test_map_relation():
    assert map_relation("operates") == "operates"
    assert map_relation("bought") == "acquired"
    assert map_relation("provides") == "offers"
    assert map_relation("underpinned by") == "enabled_by"
    assert map_relation("nonsense_xyz") == "related_to"
    print("  ✓ map_relation")


# ── Extractor tests ───────────────────────────────────────────────────────────

def test_extract_mock():
    result = extract({"id": "5", "text": "..."}, mock=True)
    assert result["chunk_id"] == "5"
    assert len(result["entities"]) > 0
    assert len(result["relations"]) > 0
    assert any(e["type"] == "Company" for e in result["entities"])
    print("  ✓ extract (mock mode, chunk 5)")


def test_extract_all_chunks_mock():
    for chunk in CHUNKS:
        result = extract(chunk, mock=True)
        assert result["chunk_id"] == chunk["id"]
    print("  ✓ extract (mock mode, all chunks)")


def test_extract_unknown_chunk_mock():
    result = extract({"id": "999", "text": "..."}, mock=True)
    assert result["chunk_id"] == "999"
    assert result["entities"] == []
    print("  ✓ extract (unknown chunk_id returns empty gracefully)")


# ── Graph builder tests ───────────────────────────────────────────────────────

def test_build_graph_basic():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)

    assert "nodes" in graph
    assert "edges" in graph
    assert "metadata" in graph
    assert graph["metadata"]["node_count"] == len(graph["nodes"])
    assert graph["metadata"]["edge_count"] == len(graph["edges"])
    print(f"  ✓ build_graph: {graph['metadata']['node_count']} nodes, "
          f"{graph['metadata']['edge_count']} edges")


def test_build_graph_has_company():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    types = {n["type"] for n in graph["nodes"]}
    assert "Company" in types
    assert "Service" in types
    assert "Event" in types
    print("  ✓ build_graph: Company, Service, Event nodes present")


def test_build_graph_no_duplicate_nodes():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids)), "Duplicate node IDs detected!"
    print("  ✓ build_graph: no duplicate node IDs")


def test_build_graph_no_duplicate_edges():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    keys = [(e["source"], e["target"], e["relation"]) for e in graph["edges"]]
    assert len(keys) == len(set(keys)), "Duplicate edges detected!"
    print("  ✓ build_graph: no duplicate edges")


def test_build_graph_provenance():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    # Every node should have at least one provenance entry
    for node in graph["nodes"]:
        assert len(node["provenance"]) >= 1, f"Node {node['id']} has no provenance"
    print("  ✓ build_graph: all nodes have provenance")


def test_build_graph_expansion():
    """Expanding an existing graph should not produce fewer nodes/edges."""
    extractions_1 = [extract(CHUNKS[0], mock=True)]
    graph_1 = build_graph(extractions_1)

    extractions_2 = [extract(c, mock=True) for c in CHUNKS[1:]]
    graph_2 = build_graph(extractions_2, existing_graph=graph_1)

    assert graph_2["metadata"]["node_count"] >= graph_1["metadata"]["node_count"]
    assert graph_2["metadata"]["edge_count"] >= graph_1["metadata"]["edge_count"]
    print(
        f"  ✓ build_graph expansion: "
        f"{graph_1['metadata']['node_count']}→{graph_2['metadata']['node_count']} nodes, "
        f"{graph_1['metadata']['edge_count']}→{graph_2['metadata']['edge_count']} edges"
    )


# ── Renderer tests ────────────────────────────────────────────────────────────

def test_render_mermaid_structure():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    mermaid = render_mermaid(graph)

    assert mermaid.startswith("graph TD"), "Must start with 'graph TD'"
    assert "-->|" in mermaid, "Must contain at least one edge"
    assert '["' in mermaid, "Must contain node label declarations"
    print("  ✓ render_mermaid: valid structure")


def test_render_mermaid_sections():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    mermaid = render_mermaid(graph)

    assert "Core Entities" in mermaid
    assert "Events" in mermaid
    print("  ✓ render_mermaid: section headers present")


def test_render_mermaid_event_ids():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    mermaid = render_mermaid(graph)

    # Event nodes should have compact IDs like E2013, E2020
    assert "E2013" in mermaid or "E2020" in mermaid, "No compact event IDs found"
    print("  ✓ render_mermaid: compact event IDs (E20xx) present")


# ── Retriever tests ───────────────────────────────────────────────────────────

def test_retrieve_returns_top_k():
    results = retrieve_chunks("food delivery products", CHUNKS, top_k=2)
    assert len(results) == 2
    for chunk, score in results:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    print("  ✓ retrieve_chunks: returns correct top_k")


def test_retrieve_ranking():
    """The chunk mentioning 'Instamart' most explicitly should rank highly."""
    results = retrieve_chunks("Instamart quick commerce grocery", CHUNKS, top_k=3)
    top_chunk = results[0][0]
    assert top_chunk["id"] in ("6", "7"), (
        f"Expected chunk 6 or 7 to be top result, got {top_chunk['id']}"
    )
    print(f"  ✓ retrieve_chunks: Instamart query → chunk {top_chunk['id']} ranked #1")


def test_retrieve_empty_query():
    results = retrieve_chunks("", CHUNKS, top_k=2)
    assert len(results) == 2
    print("  ✓ retrieve_chunks: empty query handled gracefully")


# ── Full pipeline integration test ────────────────────────────────────────────

def test_run_pipeline_mock():
    result = run_pipeline(
        CHUNKS,
        mock=True,
        query="What services does the company offer?",
        top_k=2,
    )
    assert "graph" in result
    assert "mermaid" in result
    assert "extractions" in result
    assert "retrieval" in result
    assert len(result["retrieval"]) == 2
    assert result["mermaid"].startswith("graph TD")
    print(
        f"  ✓ run_pipeline (mock, full): "
        f"{result['graph']['metadata']['node_count']} nodes, "
        f"{result['graph']['metadata']['edge_count']} edges, "
        f"{len(result['retrieval'])} retrieval results"
    )


# ── v2: Fuzzy entity normalization ───────────────────────────────────────────

def test_fuzzy_match_entity_finds_close_match():
    existing = ["Swiggy", "Instamart", "Dineout"]
    assert fuzzy_match_entity("Swiggy Limited", existing) == "Swiggy"
    assert fuzzy_match_entity("Swiggy Ltd", existing) == "Swiggy"
    print("  ✓ fuzzy_match_entity: Swiggy Limited / Swiggy Ltd → Swiggy")


def test_fuzzy_match_entity_no_false_positive():
    existing = ["Swiggy", "Instamart"]
    assert fuzzy_match_entity("Zomato", existing) is None
    print("  ✓ fuzzy_match_entity: Zomato not matched to existing nodes")


def test_fuzzy_match_entity_threshold():
    # Completely different names should not match
    existing = ["Food Delivery", "Grocery Service"]
    assert fuzzy_match_entity("Technology Capability", existing) is None
    print("  ✓ fuzzy_match_entity: unrelated names don't match")


# ── v2: Regex numeric attribute extraction ────────────────────────────────────

def test_extract_numeric_cities():
    attrs = extract_numeric_attributes("Instamart scaled to 124 cities as of March 2025")
    assert "cities" in attrs or "cities_count" in attrs
    val = attrs.get("cities") or attrs.get("cities_count")
    assert val == "124", f"Expected '124', got {val}"
    print("  ✓ extract_numeric_attributes: cities=124")


def test_extract_numeric_members():
    attrs = extract_numeric_attributes("membership base crossed 5.7 million members")
    assert "members_million" in attrs
    assert attrs["members_million"] == "5.7"
    print("  ✓ extract_numeric_attributes: members_million=5.7")


def test_extract_numeric_stake():
    attrs = extract_numeric_attributes("acquired a 100% stake in Lynk")
    assert "stake_percent" in attrs
    assert attrs["stake_percent"] == "100"
    print("  ✓ extract_numeric_attributes: stake_percent=100")


def test_extract_numeric_skus():
    attrs = extract_numeric_attributes("over 8,400 SKUs across dark stores")
    assert "skus" in attrs
    assert attrs["skus"] == "8400"
    print("  ✓ extract_numeric_attributes: skus=8400")


# ── v2: Rule-based relation extraction ───────────────────────────────────────

def test_rule_based_relations():
    rels = extract_relations_from_text("The company acquired Dineout in 2022")
    assert "acquired" in rels
    rels2 = extract_relations_from_text("Swiggy launched Instamart in 2020")
    assert "launched" in rels2
    print("  ✓ extract_relations_from_text: acquired, launched detected")


# ── v2: Structured event_type in graph ───────────────────────────────────────

def test_event_type_field():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    event_nodes = [n for n in graph["nodes"] if n["type"] == "Event"]
    assert len(event_nodes) > 0, "No event nodes found"
    for evt in event_nodes:
        assert "primary_event_type" in evt["attributes"], (
            f"Event {evt['id']} missing primary_event_type"
        )
        assert "event_types" in evt["attributes"]
        assert isinstance(evt["attributes"]["event_types"], list)
    print(f"  ✓ event nodes have structured event_type: "
          f"{set(e['attributes']['primary_event_type'] for e in event_nodes)}")


# ── v2: Hash-stable node IDs ─────────────────────────────────────────────────

def test_node_id_stability():
    """Same name+type always produces same ID."""
    id1 = make_node_id("Swiggy Instamart", "Product")
    id2 = make_node_id("Swiggy Instamart", "Product")
    assert id1 == id2, "Node IDs are not stable!"
    print(f"  ✓ node ID stable: {id1}")


def test_node_id_type_differentiation():
    """Different types produce different IDs even with same name."""
    id_cap = make_node_id("Technology", "Capability")
    id_svc = make_node_id("Technology", "Service")
    assert id_cap != id_svc
    print(f"  ✓ node IDs differ by type: {id_cap} ≠ {id_svc}")


# ── v2: Retriever backend detection ──────────────────────────────────────────

def test_retriever_backend_detected():
    backend = get_active_backend()
    assert isinstance(backend, str) and len(backend) > 0
    print(f"  ✓ active retrieval backend: {backend}")


# ── Improvement 1: Entity Type Classifier ─────────────────────────────────────

def test_classifier_product_from_context():
    etype, method = classify_entity_type(
        "Instamart",
        "In 2020, Instamart was launched in 2 cities as a quick commerce service",
    )
    assert etype == "Product", f"Expected Product, got {etype}"
    assert "rule" in method
    print(f"  ✓ classifier: 'Instamart' + launch context → {etype} ({method})")


def test_classifier_partner_from_name():
    etype, method = classify_entity_type("Restaurant Partners", "")
    assert etype == "Partner", f"Expected Partner, got {etype}"
    print(f"  ✓ classifier: 'Restaurant Partners' → {etype} ({method})")


def test_classifier_source_from_name():
    etype, method = classify_entity_type("Swiggy Limited IPO Prospectus", "")
    assert etype == "Source", f"Expected Source, got {etype}"
    print(f"  ✓ classifier: 'Swiggy Limited IPO Prospectus' → {etype} ({method})")


def test_classifier_platform_from_context():
    etype, method = classify_entity_type(
        "Consumer Convenience Platform",
        "Swiggy operates a unified convenience platform",
    )
    assert etype == "Platform", f"Expected Platform, got {etype}"
    print(f"  ✓ classifier: platform context → {etype} ({method})")


def test_classifier_keeps_llm_type_when_correct():
    """LLM-assigned type should be preserved when the rule agrees."""
    etype, method = classify_entity_type("Analytics", "", existing_type="Capability")
    assert etype == "Capability"
    assert method == "llm"
    print(f"  ✓ classifier: correct LLM type preserved → {etype} ({method})")


def test_classifier_overrides_unknown_type():
    """'Unknown' types should always be overridden by the classifier."""
    etype, method = classify_entity_type(
        "Swiggy Instamart",
        "Instamart launched in 2020",
        existing_type="Unknown",
    )
    assert etype != "Unknown", f"Expected override of Unknown, got {etype}"
    print(f"  ✓ classifier: Unknown → overridden to {etype} ({method})")


def test_classify_entities_batch():
    entities = [
        {"name": "Instamart", "type": "Unknown"},
        {"name": "Restaurant Partners", "type": "Unknown"},
        {"name": "Annual Report FY2025", "type": "Unknown"},
    ]
    result = classify_entities(entities, "Instamart was launched in 2020")
    types = {e["name"]: e["type"] for e in result}
    assert types["Restaurant Partners"] == "Partner"
    assert types["Annual Report FY2025"] == "Source"
    # All should have type_method set
    assert all("type_method" in e for e in result)
    print(f"  ✓ classify_entities batch: {types}")


def test_classifier_adds_type_method_field():
    """classify_entities must always add the type_method metadata field."""
    entities = [{"name": "Swiggy", "type": "Company"}]
    result = classify_entities(entities, "")
    assert "type_method" in result[0], "type_method field missing"
    print(f"  ✓ classifier: type_method field present → {result[0]['type_method']}")


# ── Improvement 2: Relation Confidence Scoring ────────────────────────────────

def test_edges_have_confidence_field():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    for edge in graph["edges"]:
        assert "confidence" in edge, f"Edge missing confidence: {edge}"
        assert 0.0 <= edge["confidence"] <= 1.0, f"Confidence out of range: {edge['confidence']}"
    print(f"  ✓ all {len(graph['edges'])} edges have confidence scores")


def test_edges_have_extraction_source():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    valid_sources = {"llm", "rule", "regex", "fallback"}
    for edge in graph["edges"]:
        assert "extraction_source" in edge, f"Edge missing extraction_source: {edge}"
        assert edge["extraction_source"] in valid_sources, (
            f"Unknown extraction_source: {edge['extraction_source']}"
        )
    print(f"  ✓ all edges have valid extraction_source labels")


def test_llm_edges_have_higher_confidence_than_fallback():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    llm_edges = [e for e in graph["edges"] if e.get("extraction_source") == "llm"]
    fallback_edges = [e for e in graph["edges"] if e.get("relation") == "related_to"]
    if llm_edges and fallback_edges:
        avg_llm = sum(e["confidence"] for e in llm_edges) / len(llm_edges)
        avg_fallback = sum(e["confidence"] for e in fallback_edges) / len(fallback_edges)
        assert avg_llm > avg_fallback, (
            f"LLM avg conf ({avg_llm:.2f}) should exceed fallback avg ({avg_fallback:.2f})"
        )
    print(f"  ✓ LLM edges have higher avg confidence than fallback edges")


def test_confidence_scores_loaded_from_config():
    """CONFIDENCE_SCORES should be loaded from ontology_config.json."""
    assert "llm" in CONFIDENCE_SCORES
    assert "rule" in CONFIDENCE_SCORES
    assert "regex" in CONFIDENCE_SCORES
    assert "fallback" in CONFIDENCE_SCORES
    assert CONFIDENCE_SCORES["llm"] > CONFIDENCE_SCORES["fallback"]
    print(f"  ✓ CONFIDENCE_SCORES from config: {CONFIDENCE_SCORES}")


def test_metadata_has_confidence_stats():
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    meta = graph["metadata"]
    assert "avg_edge_confidence" in meta
    assert "high_confidence_edges" in meta
    assert 0.0 <= meta["avg_edge_confidence"] <= 1.0
    print(f"  ✓ metadata confidence stats: avg={meta['avg_edge_confidence']}, "
          f"high_conf={meta['high_confidence_edges']}")


# ── Improvement 4: Ontology Config File ──────────────────────────────────────

def test_ontology_config_loads():
    """ontology_config.json should exist and load cleanly."""
    import json
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "ontology_config.json"
    assert cfg_path.exists(), f"ontology_config.json not found at {cfg_path}"
    with cfg_path.open() as f:
        cfg = json.load(f)
    assert "relations" in cfg
    assert "entity_types" in cfg
    assert "relation_synonyms" in cfg
    assert "confidence_scores" in cfg
    assert "mermaid_sections" in cfg
    print(f"  ✓ ontology_config.json loaded: "
          f"{len(cfg['relations'])} relations, {len(cfg['entity_types'])} types")


def test_ontology_constants_match_config():
    """Python constants derived from config should match the raw file."""
    import json
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "ontology_config.json"
    with cfg_path.open() as f:
        cfg = json.load(f)

    config_relations = {r["name"] if isinstance(r, dict) else r for r in cfg["relations"]}
    assert CONTROLLED_RELATIONS == config_relations, (
        f"Mismatch: config has {config_relations - CONTROLLED_RELATIONS} extra, "
        f"code has {CONTROLLED_RELATIONS - config_relations} extra"
    )
    print(f"  ✓ CONTROLLED_RELATIONS in sync with ontology_config.json "
          f"({len(CONTROLLED_RELATIONS)} relations)")


def test_metadata_has_type_counts():
    """Graph metadata should include per-type node counts."""
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    meta = graph["metadata"]
    assert "node_type_counts" in meta
    assert "event_count" in meta
    assert meta["event_count"] == meta["node_type_counts"].get("Event", 0)
    print(f"  ✓ metadata node_type_counts: {meta['node_type_counts']}")


# ── Improvement 1: Query Engine ───────────────────────────────────────────────

def _build_test_engine():
    from graphrag.query_engine import GraphQueryEngine
    extractions = [extract(c, mock=True) for c in CHUNKS]
    graph = build_graph(extractions)
    return GraphQueryEngine(graph)

def test_query_engine_products_launched_2020():
    from graphrag.query_engine import GraphQueryEngine
    engine = _build_test_engine()
    results = engine.find_products_launched(year=2020)
    names = [r["name"] for r in results]
    assert len(results) >= 1, "Expected at least one product launched in 2020"
    assert any("Instamart" in n for n in names), f"Instamart not found in {names}"
    print(f"  ✓ find_products_launched(2020): {names}")

def test_query_engine_all_products_launched():
    engine = _build_test_engine()
    results = engine.find_products_launched()
    assert len(results) >= 5, f"Expected 5+ launched products, got {len(results)}"
    years = {r["year"] for r in results}
    assert len(years) > 1, "Products should span multiple years"
    print(f"  ✓ find_products_launched(): {len(results)} results across years {sorted(years)}")

def test_query_engine_services_offered():
    engine = _build_test_engine()
    results = engine.find_services_offered()
    assert len(results) >= 3, f"Expected 3+ services, got {len(results)}"
    names = [r["name"] for r in results]
    assert any("Food" in n for n in names), f"Food Delivery not found in {names}"
    print(f"  ✓ find_services_offered(): {names}")

def test_query_engine_acquisitions():
    engine = _build_test_engine()
    results = engine.find_acquisitions()
    names = [r["name"] for r in results]
    assert "Dineout" in names, f"Dineout not in acquisitions: {names}"
    assert "Lynk" in names,    f"Lynk not in acquisitions: {names}"
    print(f"  ✓ find_acquisitions(): {names}")

def test_query_engine_capabilities():
    engine = _build_test_engine()
    results = engine.find_capabilities()
    names = [r["name"] for r in results]
    assert len(results) >= 2
    assert any("Tech" in n or "Analyt" in n or "Fulfilment" in n for n in names)
    print(f"  ✓ find_capabilities(): {names}")

def test_query_engine_events_by_type():
    engine = _build_test_engine()
    acq_events = engine.find_events_by_type("acquisition")
    assert len(acq_events) >= 1, "Expected at least one acquisition event"
    launch_events = engine.find_events_by_type("launch")
    assert len(launch_events) >= 3, "Expected 3+ launch events"
    print(f"  ✓ find_events_by_type: {len(acq_events)} acquisitions, {len(launch_events)} launches")

def test_query_engine_entities_by_type():
    engine = _build_test_engine()
    partners = engine.find_entities_by_type("Partner")
    assert len(partners) >= 3
    print(f"  ✓ find_entities_by_type('Partner'): {[p['name'] for p in partners]}")

def test_query_engine_sources():
    engine = _build_test_engine()
    sources = engine.find_sources()
    assert len(sources) >= 1
    assert all("count" in s for s in sources)
    print(f"  ✓ find_sources(): {[s['source'] for s in sources]}")

def test_query_engine_filter_by_confidence():
    engine = _build_test_engine()
    high = engine.filter_edges_by_confidence(0.85)
    low  = engine.filter_edges_by_confidence(0.0)
    assert len(high) <= len(low), "High-conf filter should return fewer edges"
    assert all(e["confidence"] >= 0.85 for e in high)
    print(f"  ✓ filter_edges_by_confidence: {len(high)} high-conf, {len(low)} total")

def test_query_engine_graph_summary():
    engine = _build_test_engine()
    s = engine.graph_summary()
    assert s["nodes"] > 0
    assert s["edges"] > 0
    assert "by_type" in s
    assert "relation_counts" in s
    assert 0.0 <= s["avg_edge_confidence"] <= 1.0
    print(f"  ✓ graph_summary(): {s['nodes']} nodes, {s['edges']} edges, "
          f"avg_conf={s['avg_edge_confidence']}")


# ── Phase 0 Regression Tests ──────────────────────────────────────────────────

def test_build_graph_source_only_no_entities():
    """
    Regression: build_graph should not raise UnboundLocalError/NameError when an
    extraction contains sources but no entities (upsert_node variable-scope fix).
    """
    extractions = [
        {
            "chunk_id": "chunk_src_only",
            "entities": [],
            "relations": [],
            "events": [],
            "sources": ["Test Regulatory Source Document"],
        }
    ]
    graph = build_graph(extractions)
    assert graph["metadata"]["node_count"] == 1
    node = graph["nodes"][0]
    assert node["type"] == "Source"
    assert node["label"] == "Test Regulatory Source Document"
    assert node["type_method"] == "rule:Source"
    assert node["provenance"] == ["chunk_src_only"]
    print("  ✓ regression: source-only extraction without entities builds correctly")


def test_build_graph_dangling_unknown_nodes():
    """
    Regression: Relations with unseen entities create Unknown nodes with fallback type_method.
    """
    extractions = [
        {
            "chunk_id": "chunk_dangling",
            "entities": [],
            "relations": [{"source": "EntityX", "target": "EntityY", "relation": "related_to"}],
            "events": [],
            "sources": [],
        }
    ]
    graph = build_graph(extractions)
    assert graph["metadata"]["node_count"] == 2
    types = {n["type"] for n in graph["nodes"]}
    assert types == {"Unknown"}
    assert all(n["type_method"] == "fallback" for n in graph["nodes"])
    print("  ✓ regression: dangling unknown nodes use fallback type_method")


def test_extractor_model_configuration():
    """
    Regression: Ensure LLM model is configurable and DEFAULT_MODEL is loaded.
    """
    from graphrag.extractor import DEFAULT_MODEL
    assert isinstance(DEFAULT_MODEL, str) and len(DEFAULT_MODEL) > 0
    result = extract({"id": "5", "text": "..."}, mock=True, model="claude-sonnet-5-5")
    assert result["chunk_id"] == "5"
    print(f"  ✓ regression: model configuration works (default: {DEFAULT_MODEL})")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        # Utils
        test_normalize_name,
        test_make_node_id,
        test_map_relation,
        # v2: Fuzzy normalization
        test_fuzzy_match_entity_finds_close_match,
        test_fuzzy_match_entity_no_false_positive,
        test_fuzzy_match_entity_threshold,
        # v2: Regex attribute extraction
        test_extract_numeric_cities,
        test_extract_numeric_members,
        test_extract_numeric_stake,
        test_extract_numeric_skus,
        # v2: Rule-based relations
        test_rule_based_relations,
        # v3 Improvement 1: Entity Type Classifier
        test_classifier_product_from_context,
        test_classifier_partner_from_name,
        test_classifier_source_from_name,
        test_classifier_platform_from_context,
        test_classifier_keeps_llm_type_when_correct,
        test_classifier_overrides_unknown_type,
        test_classify_entities_batch,
        test_classifier_adds_type_method_field,
        # v3 Improvement 2: Confidence Scoring
        test_edges_have_confidence_field,
        test_edges_have_extraction_source,
        test_llm_edges_have_higher_confidence_than_fallback,
        test_confidence_scores_loaded_from_config,
        test_metadata_has_confidence_stats,
        # Extractor
        test_extract_mock,
        test_extract_all_chunks_mock,
        test_extract_unknown_chunk_mock,
        # Graph builder
        test_build_graph_basic,
        test_build_graph_has_company,
        test_build_graph_no_duplicate_nodes,
        test_build_graph_no_duplicate_edges,
        test_build_graph_provenance,
        test_build_graph_expansion,
        # v2: Structured event_type
        test_event_type_field,
        # v2: Hash-stable IDs
        test_node_id_stability,
        test_node_id_type_differentiation,
        # Renderer
        test_render_mermaid_structure,
        test_render_mermaid_sections,
        test_render_mermaid_event_ids,
        # Retriever
        test_retrieve_returns_top_k,
        test_retrieve_ranking,
        test_retrieve_empty_query,
        # v2: Backend detection
        test_retriever_backend_detected,
        # v3 Improvement 4: Ontology config
        test_ontology_config_loads,
        test_ontology_constants_match_config,
        test_metadata_has_type_counts,
        # v3 Improvement 1: Query Engine
        test_query_engine_products_launched_2020,
        test_query_engine_all_products_launched,
        test_query_engine_services_offered,
        test_query_engine_acquisitions,
        test_query_engine_capabilities,
        test_query_engine_events_by_type,
        test_query_engine_entities_by_type,
        test_query_engine_sources,
        test_query_engine_filter_by_confidence,
        test_query_engine_graph_summary,
        # Pipeline
        test_run_pipeline_mock,
        # Phase 0 Regression tests
        test_build_graph_source_only_no_entities,
        test_build_graph_dangling_unknown_nodes,
        test_extractor_model_configuration,
    ]

    print("\n" + "=" * 55)
    print(" Graph RAG Pipeline – Test Suite")
    print("=" * 55)

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 55)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 55 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
