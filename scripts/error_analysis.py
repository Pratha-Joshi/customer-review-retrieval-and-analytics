"""Create an auditable error-analysis worksheet from evaluated query records."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluated',required=True,help='JSONL records containing query, method, metrics, and retrieved_ids');p.add_argument('--output',default='artifacts/reports/error_analysis.csv');a=p.parse_args()
 fields=['query','method','recall_at_5','observed_behavior','failure_category','notes']
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',newline='',encoding='utf-8') as output:
  writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
  for line in Path(a.evaluated).read_text().splitlines():
   row=json.loads(line); recall=row.get('metrics',{}).get('recall@5',1)
   if not recall: writer.writerow({'query':row['query'],'method':row.get('method','unknown'),'recall_at_5':recall,'observed_behavior':'No judged evidence in top 5','failure_category':'UNREVIEWED','notes':'Assign one controlled taxonomy label after manual inspection.'})
 print(f'Wrote review worksheet: {a.output}')
if __name__=='__main__':main()
