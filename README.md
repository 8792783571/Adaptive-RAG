# Adaptive RAG Inference System

## Overview
This project implements an adaptive Retrieval-Augmented Generation (RAG) system that dynamically adjusts retrieval strategies at inference time.

## Architecture
Query → Adaptive Layer → Hybrid Retriever → Re-ranker → Generator → Response

## Features
- Dynamic Top-K retrieval
- Hybrid search (vector + keyword)
- Re-ranking
- Adaptive feedback loop

## Design Decisions
- FAISS for fast similarity search
- Rule-based adaptation (no training required)

## Trade-offs
- Simplicity vs optimal ML-based adaptation
- Latency vs context depth

## Performance
- Measures latency (P50/P95)
- Tracks retrieval vs generation time

## Run
```bash
pip install -r requirements.txt
python main.py
