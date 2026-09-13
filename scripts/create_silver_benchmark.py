"""Create a deterministic, corpus-grounded *silver* known-review benchmark.

This does not claim human relevance judgments. Each query is derived from a human-written
review title, and the target review is its automatically grounded evidence document.
It is useful for reproducible regression/ablation experiments, but title overlap may favor
lexical retrieval; report that limitation with every result.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl

def category(document: dict) -> str:
    text = document['text'].lower(); rating = document['metadata']['rating']
    if any(word in text for word in ('package', 'shipping', 'delivery', 'arrived', 'sealed')): return 'complaint_analysis' if rating <= 3 else 'positive_feature'
    if any(word in text for word in ('taste', 'flavor', 'bitter', 'smell', 'texture')): return 'aspect_analysis'
    return 'complaint_analysis' if rating <= 2 else 'positive_feature' if rating >= 4 else 'thematic'

def question(title: str, rating: float) -> str:
    if rating <= 2: return f'What negative customer feedback is summarized as "{title}"?'
    if rating >= 4: return f'What positive customer feedback is summarized as "{title}"?'
    return f'What customer feedback is summarized as "{title}"?'

def eligible(document: dict) -> bool:
    title = document['metadata'].get('review_title', '').strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
    return 3 <= len(words) <= 12 and 12 <= len(title) <= 100 and len(document['text']) >= 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='data/processed/reviews_dev.jsonl')
    parser.add_argument('--output', default='data/processed/benchmark_silver.jsonl')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-fraction', type=float, default=.25)
    args = parser.parse_args()
    candidates = [doc for doc in read_jsonl(args.corpus) if eligible(doc)]
    if len(candidates) < args.size: raise ValueError(f'Only {len(candidates)} eligible documents; requested {args.size}.')
    random.Random(args.seed).shuffle(candidates)
    selected, seen_titles = [], set()
    for doc in candidates:
        title = re.sub(r'\s+', ' ', doc['metadata']['review_title']).strip().lower()
        if title in seen_titles: continue
        selected.append(doc); seen_titles.add(title)
        if len(selected) == args.size: break
    if len(selected) < args.size: raise ValueError('Could not select enough unique review titles.')
    test_start = round(args.size * (1 - args.test_fraction))
    rows = []
    for i, doc in enumerate(selected):
        meta, title = doc['metadata'], re.sub(r'\s+', ' ', doc['metadata']['review_title']).strip()
        rows.append({'query_id': f'silver_{i:03d}', 'query': question(title, meta['rating']), 'category': category(doc),
            'relevant_document_ids': [doc['id']], 'expected_answer': f'The evidence is the review titled "{title}".',
            'evidence_basis': 'Automatically grounded known-review task: query deterministically derived from the review title; target is the source review.',
            'split': 'development' if i < test_start else 'test', 'judgment_status': 'automatically_generated_silver_not_human_ground_truth',
            'generation': {'method':'deterministic_title_to_question_v1','seed':args.seed,'source_review_id':meta['review_id']}})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(f'Wrote {len(rows)} automatic silver queries: {output} ({test_start} development, {len(rows)-test_start} test).')

if __name__ == '__main__': main()
