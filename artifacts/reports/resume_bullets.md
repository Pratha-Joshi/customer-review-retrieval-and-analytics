# Resume bullets

- Built an evidence-grounded customer-review retrieval and analytics system over 568,454 processed Amazon reviews using BM25, BGE dense vectors, linear hybrid fusion, reranking, metadata-aware filtering, and review-level citations.
- Evaluated four retrieval methods on a held-out 25-query automatic silver benchmark; hybrid plus cross-encoder reranking achieved 0.810 MRR and 0.880 hit rate@10, with explicit non-human-label limitations.
- Measured a CPU quality–latency tradeoff: hybrid retrieval averaged 42 ms, while reranking added 6.63 s mean latency; implemented reproducible corpus, benchmark, and metric pipelines.
