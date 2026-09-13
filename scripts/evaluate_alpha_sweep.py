"""Efficient development-only hybrid alpha sweep with batched query encoding."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
from src.retrieval.core import BM25Retriever, DenseRetriever, HybridRetriever
from src.evaluation.metrics import aggregate, query_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',default='data/processed/reviews_dev.jsonl'); p.add_argument('--embeddings',default='artifacts/indexes/review_embeddings_dev.npy'); p.add_argument('--benchmark',default='data/processed/benchmark_silver.jsonl'); p.add_argument('--model',default='BAAI/bge-small-en-v1.5'); p.add_argument('--out',default='artifacts/metrics/hybrid_alpha_sweep_development.json'); a=p.parse_args()
    docs=read_jsonl(a.corpus); rows=[json.loads(x) for x in Path(a.benchmark).read_text().splitlines() if x]; rows=[r for r in rows if r['split']=='development']
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(a.model); queries=[r['query'] for r in rows]
    vectors=model.encode(queries,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    cache=dict(zip(queries,vectors)); dense=DenseRetriever(docs,np.load(a.embeddings),lambda texts:np.asarray([cache[t] for t in texts]))
    hybrid=HybridRetriever(BM25Retriever(docs),dense); report={'split':'development','benchmark_type':'automatic_silver_title_derived','alphas':{}}
    for alpha in [0.0,.25,.5,.75,1.0]:
        metrics=[query_metrics(hybrid.search(row['query'],top_k=10,alpha=alpha),row['relevant_document_ids']) for row in rows]
        report['alphas'][str(alpha)]=aggregate(metrics)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
