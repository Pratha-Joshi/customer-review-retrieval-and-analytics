"""Build normalized embeddings and an exact CPU FAISS inner-product index."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
def main():
 p=argparse.ArgumentParser(); p.add_argument('--corpus',default='data/processed/reviews.jsonl'); p.add_argument('--model',default='BAAI/bge-small-en-v1.5'); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--index',default='artifacts/indexes/reviews.faiss'); p.add_argument('--embeddings',default='artifacts/indexes/review_embeddings.npy'); a=p.parse_args()
 from sentence_transformers import SentenceTransformer
 import faiss
 docs=read_jsonl(a.corpus); model=SentenceTransformer(a.model); vectors=model.encode([d['text'] for d in docs],batch_size=a.batch_size,normalize_embeddings=True,show_progress_bar=True).astype('float32')
 Path(a.index).parent.mkdir(parents=True,exist_ok=True); np.save(a.embeddings,vectors); idx=faiss.IndexFlatIP(vectors.shape[1]); idx.add(vectors); faiss.write_index(idx,a.index); print(f'Indexed {idx.ntotal} documents exactly.')
if __name__=='__main__':main()
