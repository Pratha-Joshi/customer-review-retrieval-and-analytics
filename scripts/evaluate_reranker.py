"""Evaluate hybrid retrieval plus cross-encoder reranking on a frozen split."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
from src.evaluation.metrics import aggregate, query_metrics
from src.retrieval.core import BM25Retriever, DenseRetriever, HybridRetriever
from src.retrieval.rerank import CrossEncoderReranker

def summary(values):
    return {'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values,95))} if values else {'mean_ms':0.,'p95_ms':0.}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',default='data/processed/reviews_dev.jsonl'); p.add_argument('--embeddings',default='artifacts/indexes/review_embeddings_dev.npy'); p.add_argument('--benchmark',default='data/processed/benchmark_silver.jsonl'); p.add_argument('--split',choices=['development','test'],default='development'); p.add_argument('--embedding-model',default='BAAI/bge-small-en-v1.5'); p.add_argument('--reranker-model',default='cross-encoder/ms-marco-MiniLM-L-6-v2'); p.add_argument('--alpha',type=float,default=.5); p.add_argument('--candidate-k',type=int,default=30); p.add_argument('--out',default='artifacts/metrics/hybrid_reranked_silver_development.json'); a=p.parse_args()
    docs=read_jsonl(a.corpus); rows=[json.loads(x) for x in Path(a.benchmark).read_text().splitlines() if x]; rows=[r for r in rows if r['split']==a.split]
    from sentence_transformers import SentenceTransformer, CrossEncoder
    encoder=SentenceTransformer(a.embedding_model); cache=dict(zip([r['query'] for r in rows],encoder.encode([r['query'] for r in rows],batch_size=32,normalize_embeddings=True,show_progress_bar=True)))
    dense=DenseRetriever(docs,np.load(a.embeddings),lambda texts:np.asarray([cache[t] for t in texts])); hybrid=HybridRetriever(BM25Retriever(docs),dense)
    reranker=CrossEncoderReranker(CrossEncoder(a.reranker_model, max_length=512))
    per_query=[]; retrieval_times=[]; rerank_times=[]
    for row in rows:
        start=time.perf_counter(); candidates=hybrid.search(row['query'],top_k=a.candidate_k,alpha=a.alpha,candidate_k=a.candidate_k); retrieval_times.append((time.perf_counter()-start)*1000)
        ranked, rerank_ms=reranker.rerank(row['query'],candidates,top_k=10); rerank_times.append(rerank_ms)
        per_query.append({**row,'method':'hybrid_reranked','metrics':query_metrics(ranked,row['relevant_document_ids']),'retrieved_ids':[r.document['id'] for r in ranked]})
    report={'method':'hybrid_reranked','split':a.split,'query_count':len(rows),'alpha':a.alpha,'candidate_k':a.candidate_k,'metrics':aggregate([x['metrics'] for x in per_query]),'latency':{'retrieval':summary(retrieval_times),'reranking':summary(rerank_times)},'by_category':{}}
    for c in sorted({r['category'] for r in per_query}): report['by_category'][c]=aggregate([r['metrics'] for r in per_query if r['category']==c])
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
