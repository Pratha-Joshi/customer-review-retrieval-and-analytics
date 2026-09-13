"""Validate BM25 construction without duplicating the corpus on disk.

The corpus JSONL is durable. Pickling rank_bm25 duplicates token lists and can exceed
the source corpus size, so this script writes a small reproducibility manifest instead.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
from src.retrieval.core import BM25Retriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='data/processed/reviews.jsonl')
    parser.add_argument('--output', default='artifacts/indexes/bm25_manifest.json')
    parser.add_argument('--k1', type=float, default=1.5)
    parser.add_argument('--b', type=float, default=0.75)
    args = parser.parse_args()
    retriever = BM25Retriever(read_jsonl(args.corpus), args.k1, args.b)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        'corpus': args.corpus, 'document_count': len(retriever.documents),
        'k1': args.k1, 'b': args.b,
        'persistence': 'BM25 is rebuilt from corpus JSONL; no redundant pickle written.'
    }, indent=2))
    print(f'Validated BM25 over {len(retriever.documents)} documents; wrote {args.output}')

if __name__ == '__main__':
    main()
