"""Human-inspectable retrieval comparison; not an evaluation benchmark."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
from src.retrieval.core import BM25Retriever, DenseRetriever, HybridRetriever, timed_search

def print_results(name, results, latency):
    print(f'\n--- {name} ({latency:.1f} ms) ---')
    for result in results:
        m = result.document['metadata']
        excerpt = result.document['text'].replace('\n', ' ')[:240]
        print(f"{result.rank}. {result.document['id']} | score={result.score:.4f} | review={m['review_id']} | product={m['product_id']} | rating={m['rating']}\n   {excerpt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--corpus', default='data/processed/reviews_dev.jsonl')
    parser.add_argument('--embeddings', default='artifacts/indexes/review_embeddings_dev.npy')
    parser.add_argument('--model', default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=.5)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    docs = read_jsonl(args.corpus)
    bm25 = BM25Retriever(docs)
    model = SentenceTransformer(args.model)
    dense = DenseRetriever(docs, np.load(args.embeddings), lambda texts: model.encode(texts, normalize_embeddings=True))
    hybrid = HybridRetriever(bm25, dense)
    for name, retriever, kwargs in [
        ('BM25', bm25, {}), ('Dense', dense, {}),
        ('Hybrid', hybrid, {'alpha': args.alpha})]:
        results, latency = timed_search(retriever, args.query, args.top_k, **kwargs)
        print_results(name, results, latency)

if __name__ == '__main__':
    main()
