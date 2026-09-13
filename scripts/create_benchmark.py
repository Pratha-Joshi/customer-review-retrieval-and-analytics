"""Validates an auditable, human-authored benchmark; it never invents relevance labels."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import read_jsonl
def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus',default='data/processed/reviews.jsonl');p.add_argument('--benchmark',required=True);a=p.parse_args()
 ids={d['id'] for d in read_jsonl(a.corpus)}; rows=[json.loads(x) for x in Path(a.benchmark).read_text().splitlines() if x]
 for row in rows:
  required={'query','category','relevant_document_ids','expected_answer','evidence_basis','split'}
  if required-set(row): raise ValueError(f'Missing fields: {required-set(row)}')
  absent=set(row['relevant_document_ids'])-ids
  if absent: raise ValueError(f"Unknown relevance IDs for {row['query']}: {absent}")
  if row['split'] not in {'development','test'}: raise ValueError('split must be development or test')
 print(f'Validated {len(rows)} corpus-grounded benchmark questions.')
if __name__=='__main__':main()
