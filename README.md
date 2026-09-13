# Customer Review Retrieval & Evidence-Based Analytics

An end-to-end information retrieval and RAG system for searching and analyzing **568K+ Amazon product reviews**, combining lexical search, semantic retrieval, hybrid retrieval, and cross-encoder reranking.

## Features

- Retrieves relevant product reviews using **BM25**
- Performs semantic search using **BGE embeddings + FAISS**
- Combines lexical and semantic retrieval with **hybrid search**
- Reranks results using a **cross-encoder**
- Evaluates retrieval performance using **Hit Rate@K, MRR, and nDCG@10**
- Provides **product-level review analytics**
- Generates **evidence-grounded responses** using a local LLM
- Analyzes **retrieval quality and latency trade-offs**

## Tech Stack

**Python, BM25, FAISS, Sentence Transformers, BGE Embeddings, Cross-Encoder, Ollama, FastAPI**

## Workflow

```text
Amazon Reviews
      ↓
Data Processing
      ↓
┌──────────────┬───────────────┐
│     BM25     │ BGE + FAISS   │
└──────────────┴───────────────┘
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Relevant Reviews
      ↓
RAG Response
