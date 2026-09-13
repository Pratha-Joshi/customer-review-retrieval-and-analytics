.PHONY: test prepare index api ui smoke judgment-sheet benchmark-silver
test:
	pytest -q
prepare:
	python3 scripts/prepare_data.py --input data/raw/Reviews.csv
prepare-dev:
	python3 scripts/prepare_data.py --input data/raw/Reviews.csv --output data/processed/reviews_dev.jsonl --summary artifacts/reports/eda_summary_dev.json --max-documents 5000
index:
	python3 scripts/build_bm25.py
	python3 scripts/build_faiss.py
index-dev:
	python3 scripts/build_bm25.py --corpus data/processed/reviews_dev.jsonl --output artifacts/indexes/bm25_dev_manifest.json
	python3 scripts/build_faiss.py --corpus data/processed/reviews_dev.jsonl --index artifacts/indexes/reviews_dev.faiss --embeddings artifacts/indexes/review_embeddings_dev.npy
smoke:
	python3 scripts/smoke_retrieval.py --query "What problems do customers report with packaging?"
judgment-sheet:
	python3 scripts/export_judgment_sheet.py --corpus data/processed/reviews_dev.jsonl --details artifacts/reports/bm25_seed_development_details.jsonl
benchmark-silver:
	python3 scripts/create_silver_benchmark.py --corpus data/processed/reviews_dev.jsonl --output data/processed/benchmark_silver.jsonl --size 100
eval-dev-dense:
	python3 scripts/evaluate_retrieval.py --method dense --corpus data/processed/reviews_dev.jsonl --embeddings artifacts/indexes/review_embeddings_dev.npy --benchmark data/processed/benchmark_silver.jsonl --split development --out artifacts/metrics/dense_silver_development.json
eval-dev-hybrid:
	python3 scripts/evaluate_retrieval.py --method hybrid --corpus data/processed/reviews_dev.jsonl --embeddings artifacts/indexes/review_embeddings_dev.npy --benchmark data/processed/benchmark_silver.jsonl --split development --out artifacts/metrics/hybrid_silver_development.json
alpha-sweep-dev:
	python3 scripts/evaluate_alpha_sweep.py
eval-dev-rerank:
	python3 scripts/evaluate_reranker.py
eval-test-bm25:
	python3 scripts/evaluate_retrieval.py --method bm25 --corpus data/processed/reviews_dev.jsonl --benchmark data/processed/benchmark_silver.jsonl --split test --out artifacts/metrics/bm25_silver_test.json
eval-test-dense:
	python3 scripts/evaluate_retrieval.py --method dense --corpus data/processed/reviews_dev.jsonl --embeddings artifacts/indexes/review_embeddings_dev.npy --benchmark data/processed/benchmark_silver.jsonl --split test --out artifacts/metrics/dense_silver_test.json
eval-test-hybrid:
	python3 scripts/evaluate_retrieval.py --method hybrid --alpha .5 --corpus data/processed/reviews_dev.jsonl --embeddings artifacts/indexes/review_embeddings_dev.npy --benchmark data/processed/benchmark_silver.jsonl --split test --out artifacts/metrics/hybrid_silver_test.json
eval-test-rerank:
	python3 scripts/evaluate_reranker.py --split test --alpha .5 --candidate-k 30 --out artifacts/metrics/hybrid_reranked_silver_test.json
api:
	python3 -m uvicorn api.main:app --reload
ui:
	python3 -m streamlit run app/streamlit_app.py
