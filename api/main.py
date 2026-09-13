from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.data.preprocess import read_jsonl
from src.retrieval.core import BM25Retriever, timed_search
from src.analytics.service import AnalyticsService
from src.generation.grounded import citations, fallback_answer
CORPUS=os.getenv('CORPUS_PATH','data/processed/reviews.jsonl')
app=FastAPI(title='Customer Review Retrieval & Evidence Analytics'); state={}
class RetrieveRequest(BaseModel): query:str; top_k:int=5; product_id:str|None=None
class QueryRequest(RetrieveRequest): pass
class AnalyticsRequest(BaseModel): product_id:str|None=None; compare_product_id:str|None=None
@app.on_event('startup')
def startup():
 if Path(CORPUS).exists():
  docs=read_jsonl(CORPUS); state.update(documents=docs,bm25=BM25Retriever(docs),analytics=AnalyticsService(docs))
@app.get('/health')
def health(): return {'status':'ok','corpus_loaded':'documents' in state,'document_count':len(state.get('documents',[]))}
def retrieve(request):
 if 'bm25' not in state: raise HTTPException(503,'Corpus is not prepared. Run scripts/prepare_data.py.')
 allowed=state['analytics'].filter_ids(product_id=request.product_id) if request.product_id else None
 results,latency=timed_search(state['bm25'],request.query,request.top_k,allowed); return results,latency
@app.post('/retrieve')
def retrieve_endpoint(request:RetrieveRequest):
 results,latency=retrieve(request); return {'documents':[r.as_dict() for r in results],'retrieval_method':'bm25','retrieval_latency_ms':latency}
@app.post('/query')
def query_endpoint(request:QueryRequest):
 results,latency=retrieve(request); return {'answer':fallback_answer(results),'citations':citations(results),'evidence':[r.as_dict() for r in results],'retrieval_method':'bm25','retrieval_latency_ms':latency,'reranking_latency_ms':0.0,'generation_latency_ms':0.0}
@app.post('/analytics')
def analytics_endpoint(request:AnalyticsRequest):
 if 'analytics' not in state: raise HTTPException(503,'Corpus is not prepared.')
 return state['analytics'].compare(request.product_id,request.compare_product_id) if request.compare_product_id else state['analytics'].summary(request.product_id)
