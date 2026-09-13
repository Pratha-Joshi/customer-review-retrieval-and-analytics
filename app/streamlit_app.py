from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.data.preprocess import read_jsonl
from src.retrieval.core import BM25Retriever, timed_search
from src.analytics.service import AnalyticsService
st.set_page_config(page_title='Review Evidence',layout='wide'); st.title('Customer Review Retrieval & Evidence Analytics')
path=Path('data/processed/reviews.jsonl')
if not path.exists(): st.warning('Run `python scripts/prepare_data.py --input data/raw/Reviews.csv` first.'); st.stop()
@st.cache_resource
def load():
 docs=read_jsonl(path); return docs,BM25Retriever(docs),AnalyticsService(docs)
docs,bm25,analytics=load(); question=st.text_input('Question about review text'); product=st.text_input('Optional ProductId filter'); top_k=st.slider('Top K',1,10,5)
if st.button('Retrieve evidence') and question:
 allowed=analytics.filter_ids(product_id=product) if product else None; results,ms=timed_search(bm25,question,top_k,allowed); st.caption(f'BM25 retrieval: {ms:.1f} ms')
 for r in results:
  st.subheader(f"[Review {r.document['metadata']['review_id']}] · score {r.score:.3f}"); st.caption(str(r.document['metadata'])); st.write(r.document['text'])
