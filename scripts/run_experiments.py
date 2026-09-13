"""Experiment manifest writer: preserves hypotheses and configurations without inventing results."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
HYPOTHESES={
 'bm25':'Lexical retrieval should perform well under close query-review wording.',
 'dense':'Dense retrieval may recover semantically related wording.',
 'hybrid':'Combining lexical and semantic signals may improve robustness.',
 'hybrid_reranked':'Reranking may improve candidate ordering at a latency cost.',
 'alpha_sweep':'The best linear fusion weight must be determined empirically.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='artifacts/metrics/experiment_manifest.json');a=p.parse_args()
 payload={'created_at':datetime.now(timezone.utc).isoformat(),'status':'pending_dataset_and_frozen_benchmark','hypotheses':HYPOTHESES,'alpha_values':[0,.25,.5,.75,1], 'required_reporting':['overall/category metrics','mean and P95 latency','corpus size','config','code revision']}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(payload,indent=2));print(a.out)
if __name__=='__main__':main()
