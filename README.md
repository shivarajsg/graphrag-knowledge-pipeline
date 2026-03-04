# GraphRAG Knowledge Graph Pipeline

A modular pipeline that converts **unstructured company documents into a structured knowledge graph** for Graph-RAG systems.

This project extracts entities, relations, events, and attributes from text chunks and builds a queryable knowledge graph with visualization and confidence scoring.

---

# Project Overview

Many enterprise documents such as:

- IPO prospectus
- annual reports
- company descriptions
- financial filings

contain valuable information but are **unstructured text**.

This system converts those documents into a **structured knowledge graph**, enabling:

- relationship discovery
- timeline analysis
- knowledge retrieval
- Graph RAG integration

---

# Pipeline Architecture

```
Text Chunks
      ↓
Entity / Relation / Event Extraction
      ↓
Knowledge Graph Construction
      ↓
Graph JSON
      ↓
Mermaid Visualization
      ↓
Graph Query Engine
```

---

# Features

### Entity Extraction
Detects entities such as:

- Company
- Product
- Service
- Partner
- Capability
- Event

Example:

```
Swiggy launched Instamart in 2020
```

Extracted:

```
Entity: Swiggy (Company)
Entity: Instamart (Product)
```

---

### Relation Extraction

Identifies relationships between entities.

Example:

```
Swiggy → launched → Instamart
Swiggy → acquired → Dineout
```

Includes:

- rule-based extraction
- fallback relations
- controlled relation vocabulary

---

### Event Modeling

Important business events are captured as nodes.

Example:

```
Event_2020_launch_instamart
```

Allows queries like:

- products launched in 2020
- acquisition timeline

---

### Entity Normalization

Prevents duplicate nodes.

Example:

```
Swiggy
Swiggy Ltd
Swiggy Limited
```

Normalized to:

```
Swiggy
```

---

### Attribute Extraction

Extracts numeric information from text.

Example:

```
Instamart expanded to 124 cities
```

Extracted:

```
cities = 124
```

---

### Confidence Scoring

Each relationship includes a confidence score.

Example:

```json
{
 "source": "swiggy",
 "relation": "acquired",
 "target": "dineout",
 "confidence": 0.9
}
```

Confidence sources:

- LLM extraction
- rule-based extraction
- regex extraction

---

### Ontology Configuration

Graph schema is configurable using:

```
ontology_config.json
```

Defines:

- entity types
- relation types
- confidence scores
- synonym mappings

---

### Source Tracking

Every entity is linked to the document chunk it came from.

```
Swiggy → described_in → Source_1
```

This improves traceability.

---

### Query Engine

The project includes a graph query engine that allows traversal queries.

Examples:

```
products launched in 2020
services offered by the company
list of acquisitions
```

Example usage:

```python
engine.find_products_launched(2020)
engine.find_services_offered()
engine.find_acquisitions()
```

---

### CLI Interface

Queries can be executed directly from terminal.

```
python main.py --query "products launched in 2020"
python main.py --query "acquisitions"
python main.py --query "services"
```

---

# Project Structure

```
graphrag_pipeline/
│
├── main.py
├── ontology_config.json
├── requirements.txt
│
├── graphrag/
│   ├── extractor.py
│   ├── classifier.py
│   ├── graph_builder.py
│   ├── query_engine.py
│   ├── renderer.py
│   ├── retriever.py
│   └── utils.py
│
├── data/
│   └── sample_chunks.json
│
├── outputs/
│   ├── graph.json
│   └── graph.mmd
│
└── tests/
    └── test_pipeline.py
```

---

# Example Graph Output

```
Swiggy → launched → Instamart
Swiggy → acquired → Dineout
Platform → offers → Food Delivery
```

---

# Visualization

Mermaid graph output:

```
graph TD
Swiggy -->|launched| Instamart
Swiggy -->|acquired| Dineout
```

To visualize:

1. Open https://mermaid.live
2. Paste contents of `graph.mmd`

---

# Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run pipeline:

```
python main.py
```

Run with queries:

```
python main.py --query "products launched in 2020"
```

---

# Test Coverage

The project includes **57 automated tests** verifying:

- extraction
- graph construction
- entity normalization
- confidence scoring
- query engine
- pipeline behavior

Run tests:

```
python -m tests.test_pipeline
```

---

# Technologies Used

- Python
- Knowledge Graph modeling
- JSON graph storage
- Regex extraction
- Rule-based NLP
- Mermaid graph visualization

---

# Future Improvements

Potential extensions:

- Neo4j graph database integration
- LLM-powered extraction
- Graph embeddings
- GraphRAG integration with LLMs

---

# Author

Shivraj  
Computer Science Student | AI / ML Enthusiast
