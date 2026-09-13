"""Evaluate BM25 against frozen development or held-out test questions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
from src.retrieval.core import BM25Retriever, DenseRetriever, HybridRetriever
from src.evaluation.metrics import query_metrics,aggregate
def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus',default='data/processed/reviews.jsonl');p.add_argument('--benchmark',default='data/processed/benchmark.jsonl');p.add_argument('--split',choices=['development','test'],default='test');p.add_argument('--method',choices=['bm25','dense','hybrid'],default='bm25');p.add_argument('--embeddings');p.add_argument('--model',default='BAAI/bge-small-en-v1.5');p.add_argument('--alpha',type=float,default=.5);p.add_argument('--out',default='artifacts/metrics/bm25.json');p.add_argument('--details-out',help='Optional JSONL per-query rankings for manual error analysis.');a=p.parse_args()
 docs=read_jsonl(a.corpus); bm25=BM25Retriever(docs)
 if a.method == 'bm25': retriever=bm25
 else:
  if not a.embeddings: p.error('--embeddings is required for dense or hybrid evaluation')
  from sentence_transformers import SentenceTransformer
  model=SentenceTransformer(a.model); dense=DenseRetriever(docs,np.load(a.embeddings),lambda texts:model.encode(texts,normalize_embeddings=True))
  retriever=dense if a.method == 'dense' else HybridRetriever(bm25,dense)
 rows=[json.loads(x) for x in Path(a.benchmark).read_text().splitlines() if x]; rows=[r for r in rows if r['split']==a.split]
 evaluated=[]
 for row in rows:
  kwargs={'alpha':a.alpha} if a.method == 'hybrid' else {}
  result=retriever.search(row['query'],top_k=10,**kwargs); metrics=query_metrics(result,row['relevant_document_ids']); evaluated.append({**row,'method':a.method,'metrics':metrics,'retrieved_ids':[r.document['id'] for r in result]})
 report={'method':a.method,'split':a.split,'query_count':len(evaluated),'metrics':aggregate([x['metrics'] for x in evaluated]),'by_category':{}}
 for category in sorted({x['category'] for x in evaluated}): report['by_category'][category]=aggregate([x['metrics'] for x in evaluated if x['category']==category])
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2))
 if a.details_out:
  details=Path(a.details_out);details.parent.mkdir(parents=True,exist_ok=True)
  details.write_text(''.join(json.dumps(row)+'\n' for row in evaluated))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
