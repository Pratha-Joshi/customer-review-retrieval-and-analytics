from __future__ import annotations
import re, time
from dataclasses import dataclass
from typing import Callable
import numpy as np
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]: return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
@dataclass
class SearchResult:
    document: dict; score: float; rank: int
    def as_dict(self): return {"id": self.document["id"], "score": self.score, "rank": self.rank, "metadata": self.document["metadata"], "text": self.document["text"]}

class BM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents; self.index = BM25Okapi([tokenize(d["text"]) for d in documents], k1=k1, b=b)
    def search(self, query, top_k=5, allowed_ids=None):
        scores = self.index.get_scores(tokenize(query)); idx = np.argsort(scores)[::-1]
        if allowed_ids is not None: idx = [i for i in idx if self.documents[i]["id"] in allowed_ids]
        return [SearchResult(self.documents[i], float(scores[i]), rank + 1) for rank, i in enumerate(idx[:top_k])]

class DenseRetriever:
    def __init__(self, documents, embeddings, encoder: Callable[[list[str]], np.ndarray]):
        self.documents, self.embeddings, self.encoder = documents, self._normalize(embeddings), encoder
    @staticmethod
    def _normalize(x):
        x=np.asarray(x, dtype=np.float32); return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    def search(self, query, top_k=5, allowed_ids=None):
        q=self._normalize(self.encoder([query]))[0]; scores=self.embeddings @ q; idx=np.argsort(scores)[::-1]
        if allowed_ids is not None: idx=[i for i in idx if self.documents[i]["id"] in allowed_ids]
        return [SearchResult(self.documents[i], float(scores[i]), rank+1) for rank,i in enumerate(idx[:top_k])]

def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores: return {}
    lo, hi = min(scores.values()), max(scores.values())
    return {key: (value-lo)/(hi-lo) if hi > lo else 1.0 for key,value in scores.items()}

class HybridRetriever:
    def __init__(self, bm25, dense): self.bm25, self.dense = bm25, dense
    def search(self, query, top_k=5, alpha=.5, candidate_k=30, allowed_ids=None):
        sparse=self.bm25.search(query, candidate_k, allowed_ids); dense=self.dense.search(query, candidate_k, allowed_ids)
        docs={r.document["id"]: r.document for r in sparse+dense}; bs=minmax({r.document['id']:r.score for r in sparse}); ds=minmax({r.document['id']:r.score for r in dense})
        fused={i: alpha*bs.get(i,0)+(1-alpha)*ds.get(i,0) for i in docs}
        return [SearchResult(docs[i], score, rank+1) for rank,(i,score) in enumerate(sorted(fused.items(), key=lambda x:x[1], reverse=True)[:top_k])]

def timed_search(retriever, *args, **kwargs):
    start=time.perf_counter(); results=retriever.search(*args, **kwargs); return results, (time.perf_counter()-start)*1000
