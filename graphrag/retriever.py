"""
retriever.py - v2
-----------------
Backend priority: sentence-transformers → sklearn TF-IDF → pure-Python TF-IDF
"""
from __future__ import annotations
import math, re, logging
from collections import Counter
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

def retrieve_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
    if not chunks: return []
    if not query.strip(): return [(c, 0.0) for c in chunks[:top_k]]
    for fn in [_semantic_retrieve, _sklearn_retrieve, _pure_python_retrieve]:
        try: return fn(query, chunks, top_k)
        except ImportError: continue
        except Exception as e: logger.warning(f"[retriever] {fn.__name__} failed: {e}"); continue
    return []

def get_active_backend() -> str:
    try: import sentence_transformers; return "sentence-transformers (semantic)"
    except ImportError: pass
    try: import sklearn; return "scikit-learn TF-IDF"
    except ImportError: pass
    return "pure-Python TF-IDF"

def _semantic_retrieve(query, chunks, top_k):
    """Embedding cosine similarity via sentence-transformers (all-MiniLM-L6-v2)."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c.get("text", "") for c in chunks]
    embs = model.encode([query] + texts, convert_to_numpy=True, show_progress_bar=False)
    q_emb, d_embs = embs[0], embs[1:]
    def cos(a, b):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d else 0.0
    scores = [cos(q_emb, d) for d in d_embs]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = [(chunks[i], float(s)) for i, s in ranked[:top_k]]
    logger.info(f"[retriever:semantic] scores={[f'{s:.3f}' for _,s in results]}")
    return results

def _sklearn_retrieve(query, chunks, top_k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [c.get("text", "") for c in chunks]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)
    tfidf = vec.fit_transform([query] + texts)
    scores = cosine_similarity(tfidf[0], tfidf[1:]).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = [(chunks[i], float(s)) for i, s in ranked[:top_k]]
    logger.info(f"[retriever:sklearn] scores={[f'{s:.3f}' for _,s in results]}")
    return results

_STOP = {"a","an","the","is","it","in","of","to","and","or","for","with","as","at","by","from",
         "on","that","this","its","are","was","were","be","been","have","has","had","will","would",
         "could","should","may","might","not","but","if","so","also","their","they","which","into"}

def _tok(text): return [t for t in re.findall(r"[a-zA-Z]+", text.lower()) if t not in _STOP and len(t)>1]
def _tf(toks): c=Counter(toks); n=len(toks); return {t:v/n for t,v in c.items()} if n else {}
def _idf(corpus): N=len(corpus); df=Counter(t for doc in corpus for t in set(doc)); return {t:math.log((N+1)/(v+1))+1 for t,v in df.items()}
def _tfidf(toks,idf): tf=_tf(toks); return {t:v*idf.get(t,0) for t,v in tf.items()}
def _cos(a,b):
    common=set(a)&set(b); dot=sum(a[t]*b[t] for t in common)
    ma=math.sqrt(sum(v**2 for v in a.values())); mb=math.sqrt(sum(v**2 for v in b.values()))
    return dot/(ma*mb) if ma and mb else 0.0

def _pure_python_retrieve(query, chunks, top_k):
    texts=[c.get("text","") for c in chunks]
    corpus_toks=[_tok(t) for t in [query]+texts]
    idf=_idf(corpus_toks)
    qv=_tfidf(corpus_toks[0],idf); dvs=[_tfidf(t,idf) for t in corpus_toks[1:]]
    scores=[_cos(qv,d) for d in dvs]
    ranked=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
    results=[(chunks[i],s) for i,s in ranked[:top_k]]
    logger.info(f"[retriever:pure-python] scores={[f'{s:.3f}' for _,s in results]}")
    return results
