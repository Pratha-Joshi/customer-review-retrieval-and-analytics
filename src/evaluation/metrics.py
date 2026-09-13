from __future__ import annotations
import math
def _ids(results): return [r.document['id'] if hasattr(r, 'document') else r['id'] for r in results]
def query_metrics(results, relevant_ids, ks=(1,3,5,10)):
    rel=set(relevant_ids); ranked=_ids(results); out={}
    for k in ks:
        hits=[d in rel for d in ranked[:k]]
        # Standard recall measures the fraction of all judged evidence recovered.
        # `hit_rate@k` retains the useful at-least-one-evidence interpretation.
        out[f'recall@{k}']=sum(hits)/len(rel) if rel else 0.0
        out[f'hit_rate@{k}']=float(any(hits))
        out[f'precision@{k}']=sum(hits)/k
        dcg=sum(hit/math.log2(i+2) for i,hit in enumerate(hits)); ideal=sum(1/math.log2(i+2) for i in range(min(len(rel),k)))
        out[f'ndcg@{k}']=dcg/ideal if ideal else 0.0
    out['mrr']=next((1/(i+1) for i,d in enumerate(ranked) if d in rel),0.0)
    return out
def aggregate(rows):
    keys=sorted({k for row in rows for k in row}); return {k:sum(row.get(k,0) for row in rows)/len(rows) for k in keys} if rows else {}
