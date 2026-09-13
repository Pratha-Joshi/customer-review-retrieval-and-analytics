"""Export pooled retrieval candidates for human relevance adjudication."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--details', required=True)
    parser.add_argument('--output', default='artifacts/reports/relevance_judgments.csv')
    args = parser.parse_args()
    docs = {doc['id']: doc for doc in read_jsonl(args.corpus)}
    fields = ['query', 'category', 'document_id', 'review_id', 'product_id', 'rating', 'rank', 'text', 'relevance_label', 'reviewer_notes']
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for line in Path(args.details).open(encoding='utf-8'):
            record = json.loads(line)
            for rank, document_id in enumerate(record['retrieved_ids'], 1):
                doc = docs[document_id]; meta = doc['metadata']
                writer.writerow({'query': record['query'], 'category': record['category'], 'document_id': document_id,
                    'review_id': meta['review_id'], 'product_id': meta['product_id'], 'rating': meta['rating'],
                    'rank': rank, 'text': doc['text'], 'relevance_label': '', 'reviewer_notes': ''})
    print(f'Wrote {output}; assign relevance_label 2 (direct), 1 (partial), or 0 (not relevant).')

if __name__ == '__main__':
    main()
