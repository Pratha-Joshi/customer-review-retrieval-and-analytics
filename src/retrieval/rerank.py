from __future__ import annotations
import time
from .core import SearchResult
class CrossEncoderReranker:
    def __init__(self, model): self.model=model
    def rerank(self, query, candidates, top_k=5):
        start=time.perf_counter(); scores=self.model.predict([(query, c.document['text']) for c in candidates])
        ranked=sorted(zip(candidates, scores), key=lambda x:x[1], reverse=True)[:top_k]
        return [SearchResult(c.document, float(s), i+1) for i,(c,s) in enumerate(ranked)], (time.perf_counter()-start)*1000
