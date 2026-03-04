# Graph RAG Pipeline  ·  v2

A reusable, production-grade pipeline that converts raw text chunks into a
**Graph RAG-ready knowledge graph** — producing structured JSON and a
paste-ready Mermaid diagram.

---

## Pipeline Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        INPUT                                    │
  │              chunks  (JSON array, any company)                  │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 1 · extract()           graphrag/extractor.py           │
  │                                                                 │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
  │   │  LLM mode    │ OR │  Mock mode   │    │ Rule-based   │    │
  │   │ (Anthropic)  │    │ (offline)    │    │ relation map │    │
  │   └──────────────┘    └──────────────┘    └──────────────┘    │
  │                                                                 │
  │   Output per chunk:  entities · relations · events · sources   │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 2 · build_graph()       graphrag/graph_builder.py       │
  │                                                                 │
  │   ┌──────────────────┐  ┌──────────────────┐                  │
  │   │ Fuzzy entity     │  │ Regex numeric     │                  │
  │   │ normalization    │  │ attr extraction   │                  │
  │   │ (rapidfuzz /     │  │ (cities, members, │                  │
  │   │  difflib)        │  │  years, SKUs …)   │                  │
  │   └──────────────────┘  └──────────────────┘                  │
  │                                                                 │
  │   Stable node IDs · dedup · provenance · event_type tagging    │
  │                                                                 │
  │   Output:  Graph JSON  { nodes · edges · metadata }            │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
  ┌────────────────────────┐  ┌─────────────────────────────────────┐
  │ STAGE 3                │  │ STAGE 4 (optional)                  │
  │ render_mermaid()       │  │ retrieve_chunks()                   │
  │ graphrag/renderer.py   │  │ graphrag/retriever.py               │
  │                        │  │                                     │
  │ graph TD               │  │ ┌─────────────────────┐            │
  │  sections + edges      │  │ │sentence-transformers│ ← best     │
  │  described_in links    │  │ │  (semantic search)  │            │
  │  event timeline        │  │ └─────────────────────┘            │
  │                        │  │ ┌─────────────────────┐            │
  └────────────────────────┘  │ │  sklearn TF-IDF     │ ← fallback │
                              │ └─────────────────────┘            │
                              │ ┌─────────────────────┐            │
                              │ │  pure-Python TF-IDF │ ← always   │
                              │ └─────────────────────┘            │
                              └─────────────────────────────────────┘
```

---

## Module map

| File | Function | Role |
|------|----------|------|
| `graphrag/extractor.py` | `extract()` | LLM/mock extraction → entities, relations, events, sources |
| `graphrag/graph_builder.py` | `build_graph()` | Assembles + expands graph JSON with fuzzy dedup & provenance |
| `graphrag/renderer.py` | `render_mermaid()` | Graph JSON → `graph TD` Mermaid text |
| `graphrag/retriever.py` | `retrieve_chunks()` | Semantic or TF-IDF chunk ranking |
| `graphrag/pipeline.py` | `run_pipeline()` | Orchestrates all four stages |
| `graphrag/ontology.py` | — | Controlled relation vocab, node types, synonym map |
| `graphrag/utils.py` | — | Fuzzy normalization, stable IDs, regex attrs, dedup helpers |
| `graphrag/mock_data.py` | — | Pre-extracted data for the 3 Swiggy test chunks |

---

## What's new in v2

| # | Improvement | Where |
|---|-------------|-------|
| 1 | **Fuzzy entity normalization** — `rapidfuzz` (falls back to `difflib`) prevents duplicates like "Swiggy" / "Swiggy Limited" / "Swiggy Ltd" | `utils.fuzzy_match_entity()` |
| 2 | **Regex numeric attribute extraction** — auto-extracts `cities=124`, `members_million=5.7`, `stake_percent=100`, etc. from text | `utils.extract_numeric_attributes()` |
| 3 | **Rule-based relation backup** — keyword patterns fire when LLM is uncertain | `utils.extract_relations_from_text()` |
| 4 | **Structured event `event_type` field** — events carry both raw tags and a clean `primary_event_type` (launch / acquisition / milestone / fundraise / listing / corporate) | `graph_builder._classify_event_types()` |
| 5 | **Stable hash-based node IDs** — IDs include a 6-char MD5 hash to survive name variations across expansions | `utils.make_node_id()` |
| 6 | **Semantic retrieval** — `sentence-transformers` backend auto-selected when installed; graceful fallback chain otherwise | `retriever.py` |
| 7 | **Architecture diagram** in README (this file) | — |

Items already present in v1: source nodes, `described_in` edges, relation fallback (`related_to`), Mermaid sections, multi-chunk provenance tracking, graph expansion.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Minimum install** (no API, no GPU, zero optional deps):
```bash
pip install anthropic              # only if you want LLM mode
```
Everything else degrades gracefully.

---

## Usage

### Mock mode — works immediately, no API key needed
```bash
python main.py --chunks data/sample_chunks.json --mock
```

### LLM mode
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --chunks data/sample_chunks.json
```

### With retrieval query
```bash
python main.py --chunks data/sample_chunks.json --mock \
  --query "What products does the company offer?" --top-k 3
```

### Expand an existing graph with new chunks
```bash
python main.py --chunks data/new_chunks.json --mock \
  --existing outputs/graph.json
```

---

## Example input → output

**Input** (`data/sample_chunks.json`, chunk 7 excerpt):
```
"In 2022, the company acquired and integrated Dineout, expanded Swiggy
 Instamart to cover 25 cities with over 400 dark stores and over 8,400 SKUs…"
```

**Output node** (graph.json):
```json
{
  "id": "product_swiggy_instamart_a3f9c1",
  "type": "Product",
  "label": "Swiggy Instamart",
  "attributes": {
    "launch_year": "2020",
    "cities": "124",
    "dark_stores": "400",
    "skus": "8400",
    "scale_2025": "124 cities"
  },
  "provenance": ["6", "7"]
}
```

**Output event node**:
```json
{
  "id": "E2022",
  "type": "Event",
  "attributes": {
    "year": "2022",
    "primary_event_type": "acquisition",
    "event_types": ["acquisition", "launch", "milestone"],
    "tags": ["Acquisition", "Launch", "Scale"]
  }
}
```

**Output Mermaid** (paste into [mermaid.live](https://mermaid.live)):
```
graph TD
    %% Core Entities
    company_swiggy["Company: Swiggy<br/>..."]
    ...
    E2022["Event: Event 2022 (2022)<br/>..."]
    company_swiggy -->|has_event| E2022
    E2022 -->|acquired| acquisition_dineout
```

---

## Example Queries

The graph is designed to answer natural-language questions through structured
traversal. Each query maps to a graph path you can follow in `graph.json` or
visualise in the Mermaid diagram.

---

### Q1 — What services does the company offer?

**Graph traversal:**
```
Company  →[operates]→  Platform  →[offers]→  Service
```

**Answer from graph:**
- Food Delivery
- Dining Out and Events
- Grocery and Household Items
- Other Hyperlocal Services

**How to find it in graph.json:**
```python
# Find edges where relation == "operates" from a Company node
# then follow "offers" edges from the resulting Platform node
company_node   = [n for n in graph["nodes"] if n["type"] == "Company"][0]
platform_edges = [e for e in graph["edges"]
                  if e["source"] == company_node["id"] and e["relation"] == "operates"]
service_edges  = [e for e in graph["edges"]
                  if e["source"] == platform_edges[0]["target"] and e["relation"] == "offers"]
```

---

### Q2 — What products were launched in 2020?

**Graph traversal:**
```
Company  →[has_event]→  E2020  →[launched]→  Product
```

**Answer from graph:**
- Swiggy Instamart
- Swiggy Genie

**How to find it in graph.json:**
```python
event_2020 = next(n for n in graph["nodes"]
                  if n["type"] == "Event" and n["attributes"]["year"] == "2020")
launched   = [e for e in graph["edges"]
              if e["source"] == event_2020["id"] and e["relation"] == "launched"]
products   = [n for n in graph["nodes"]
              if n["id"] in {e["target"] for e in launched}]
```

---

### Q3 — What companies did the company acquire?

**Graph traversal:**
```
Company  →[acquired]→  Acquisition
```

**Answer from graph:**
- Dineout  (acquired_year=2022)
- Lynk  (acquired_year=2023, stake=100%)

**How to find it in graph.json:**
```python
acquisitions = [e for e in graph["edges"] if e["relation"] == "acquired"]
targets = [n for n in graph["nodes"]
           if n["id"] in {e["target"] for e in acquisitions}]
```

---

### Q4 — What capabilities power the platform?

**Graph traversal:**
```
Platform  →[enabled_by]→  Capability
```

**Answer from graph:**
- Technology  (reusable tech stack)
- Analytics
- Fulfilment

---

### Q5 — Which edges should I trust most?

Filter by confidence score in `graph.json`:

```python
high_confidence = [e for e in graph["edges"] if e["confidence"] >= 0.75]
low_confidence  = [e for e in graph["edges"] if e["confidence"] < 0.75]
```

Confidence is assigned by extraction method:

| Source | Score | Meaning |
|--------|-------|---------|
| `llm` | 0.90 | LLM reasoned about this relation |
| `rule` | 0.75 | Controlled vocab / rule match |
| `regex` | 0.65 | Pattern-based extraction |
| `fallback` | 0.50 | `related_to` catch-all |

---

### Q6 — What is the full product timeline?

**Graph traversal — all events in year order:**
```
Company  →[has_event]→  E20XX  →[launched/acquired]→  Product/Acquisition
```

```python
events = sorted(
    [n for n in graph["nodes"] if n["type"] == "Event"],
    key=lambda n: n["attributes"]["year"]
)
for evt in events:
    year  = evt["attributes"]["year"]
    etype = evt["attributes"]["primary_event_type"]
    desc  = evt["attributes"]["description"][:60]
    print(f"{year}  [{etype}]  {desc}")
```

---

### Q7 — Where is the information sourced from?

**Graph traversal:**
```
Entity  →[described_in]→  Source
```

```python
sources = [n for n in graph["nodes"] if n["type"] == "Source"]
for src in sources:
    entities = [e["source"] for e in graph["edges"]
                if e["target"] == src["id"] and e["relation"] == "described_in"]
    print(f"{src['label']}: {len(entities)} entities")
```

---

| Relation | Example |
|----------|---------|
| `operates` | Company operates Platform |
| `offers` | Platform offers Service |
| `enabled_by` | Platform enabled_by Capability |
| `supported_by` | Platform supported_by Partners |
| `includes` | Programme includes Product |
| `integrated_surface` | Product integrated with Product |
| `executed_via` | Service executed_via Product |
| `acquired` | Company acquired Target |
| `launched` | Event launched Product |
| `has_event` | Company has_event Event |
| `benefits_across` | Programme benefits_across Services |
| `feature` | Service feature Feature |
| `described_in` | Entity described_in Source |
| `related_to` | Fallback — used sparingly |

---

## Chunk format

```json
[
  {
    "id":   "unique_chunk_id",
    "text": "Full text of the chunk …"
  }
]
```

---

## Running tests

```bash
python tests/test_pipeline.py
# or
python -m pytest tests/ -v
```

All tests run in mock mode — no API key or internet required.

---

## Adding new relation types

Edit `graphrag/ontology.py` only:
1. Add to `CONTROLLED_RELATIONS`
2. Add synonyms to `RELATION_SYNONYM_MAP`
3. Add to `ONTOLOGY_DESCRIPTION` (injected into LLM prompt)
