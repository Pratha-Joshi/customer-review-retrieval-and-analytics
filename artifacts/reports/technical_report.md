# Technical report: Customer Review Retrieval & Evidence-Based Analytics

## Problem formulation

The system retrieves review text that supports an open-ended customer-feedback question and presents the evidence alongside any synthesis. Structured questions (for example, mean rating) are answered with aggregation, not reported as text-retrieval outcomes.

## Data and representation

The pipeline inspects the downloaded CSV rather than presuming a schema. It preserves source text and metadata, makes stable content-addressed review document IDs, drops only empty/unusable review records, and uses whole reviews as documents. This is CPU-oriented and avoids accidental context loss from universal chunking.

## Methods and hypotheses

BM25 tests lexical overlap; normalized BGE/FAISS tests semantic matching; a linear normalized score fusion tests whether combining both helps; MiniLM cross-encoder reranking tests whether candidate ordering improves. The hypotheses are directional possibilities, not conclusions.

## Evaluation methodology

Questions must carry an evidence basis and document-level relevance IDs, be split before tuning, and be validated against the corpus. Report Recall@1/3/5/10, precision@K, MRR, and nDCG@5/10 overall and by category. Generation evaluation is separate: assess relevance, grounding, context relevance, and citation correctness against the displayed evidence.

## Results

The full raw corpus contains 568,454 processed reviews. CPU-oriented retrieval experiments used a deterministic 5,000-review development corpus and a 100-question automatically generated silver benchmark (75 development, 25 held-out test). Its queries are derived from human-written review titles and target their source review. This makes it reproducible, but creates lexical overlap and does **not** constitute human relevance ground truth.

After selecting linear fusion α=0.50 on development (MRR 0.693), the held-out test results were:

| Method | Hit rate@1 | Hit rate@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.600 | 0.760 | 0.659 | 0.684 |
| Dense | 0.480 | 0.720 | 0.552 | 0.592 |
| Hybrid | 0.600 | 0.800 | 0.672 | 0.703 |
| Hybrid + cross-encoder | 0.760 | 0.880 | 0.810 | 0.828 |

The cross-encoder produced the strongest ordering on this diagnostic, but on the CPU laptop hybrid retrieval averaged 42.4 ms (P95 69.1 ms) and reranking added 6,627.6 ms mean latency (P95 10,440.3 ms). The project therefore treats reranking as an optional high-quality mode. The result does not establish that reranking is optimal for broad customer-insight questions; that requires pooled or human relevance judgments.

## Error analysis protocol

For weak results, label lexical mismatch, semantic mismatch, insufficient evidence, redundancy, contradiction, ambiguity, multi-intent question, rare terminology, short review, metadata dependency, or relevance-label issue. Aggregate metrics do not replace inspecting these examples.

## Tradeoffs and limitations

Exact FAISS search is simple and auditable but may not scale indefinitely. The local generator is optional because an answer without evidence is not a successful result. Automatic relevance labels are not human ground truth; representative manual review is needed before defensible benchmark claims.

## Conclusion

This project is designed to empirically determine method tradeoffs, rather than encoding an assumption that hybrid retrieval or reranking wins.
