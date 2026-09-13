# Interview questions

1. **Why BM25?** A strong transparent lexical baseline; it exposes exact term-match behavior.
2. **Why dense retrieval?** It can surface semantically similar phrasing with low lexical overlap.
3. **Why hybrid?** It tests complementary signals using explicit linear score fusion.
4. **Why rerank?** A cross-encoder jointly reads query and candidate, trading latency for ordering quality.
5. **Why FAISS?** It provides efficient local vector similarity search; this project uses exact `IndexFlatIP` for auditability.
6. **Why normalize embeddings?** Inner product of unit vectors equals cosine similarity.
7. **What does BM25 k1 control?** Term-frequency saturation.
8. **What does BM25 b control?** Document-length normalization strength.
9. **How are scores combined?** Candidate scores are min–max normalized within each retrieval list and linearly weighted by alpha.
10. **Why is this not RRF?** RRF combines ranks via reciprocal functions, not raw normalized scores.
11. **Why not simply use an LLM?** It cannot guarantee review evidence retrieval or make its claims auditable.
12. **Why separate metadata and text?** Rating/count aggregation and unstructured evidence are distinct tasks with distinct failure modes.
13. **What is Recall@K here?** Whether at least one judged evidence document appears in the first K results.
14. **What is MRR?** Reciprocal rank of the first relevant result, averaged over queries.
15. **What is nDCG?** A position-sensitive relevance metric normalized by an ideal ranking.
16. **Can high recall still yield a bad answer?** Yes—redundant, contradictory, or poorly cited context can harm synthesis.
17. **How is leakage prevented?** Hyperparameters are tuned only on development questions; test is frozen.
18. **How are relevance labels audited?** Every question records its evidence basis and IDs are checked against the corpus.
19. **What is a key benchmark limitation?** Automatically generated labels are synthetic and cannot be called human ground truth.
20. **How are product-specific complaints handled?** Filter product metadata first, then retrieve text from that subset.
21. **Why whole reviews?** Customer reviews are often short; chunking can destroy complete supporting statements.
22. **What does reranking latency include?** Candidate-pair cross-encoder scoring only, separately from first-stage retrieval.
23. **What causes lexical failure?** Synonyms, paraphrases, spelling variants, and rare terms.
24. **What causes dense failure?** Semantic overgeneralization or weak representations of product-specific wording.
25. **How do you handle contradictions?** Retrieve and show both sides; prompt generation to label disagreement.
26. **How do new documents enter the system?** Re-run preprocessing and rebuild exact indexes, or adopt incremental indexing with versioning.
27. **What changes at production scale?** Approximate vector search, queues, observability, access controls, and continual evaluation.
28. **How monitor quality?** Track latency, citation clicks, judged retrieval metrics, and error categories over time.
29. **Why avoid LangChain/LlamaIndex?** Direct code keeps scoring, metadata filters, and evaluation behavior inspectable.
30. **What did you learn?** A measured comparison is more defensible than assuming a more complex method is better.
