"""
graphrag – A reusable Graph RAG pipeline mini-library.

Public API
----------
    from graphrag.extractor    import extract
    from graphrag.graph_builder import build_graph
    from graphrag.renderer     import render_mermaid
    from graphrag.retriever    import retrieve_chunks
    from graphrag.pipeline     import run_pipeline
"""

__version__ = "3.0.0"

from .extractor      import extract
from .graph_builder  import build_graph
from .renderer       import render_mermaid
from .retriever      import retrieve_chunks
from .pipeline       import run_pipeline
from .classifier     import classify_entity_type, classify_entities
from .query_engine   import GraphQueryEngine, load_engine

__all__ = [
    "extract",
    "build_graph",
    "render_mermaid",
    "retrieve_chunks",
    "run_pipeline",
    "classify_entity_type",
    "classify_entities",
    "GraphQueryEngine",
    "load_engine",
]
