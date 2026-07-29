# System Design RAG

A work-in-progress Retrieval-Augmented Generation application that answers
system-design questions from a PDF document.

## Current pipeline

- Loads and cleans a PDF
- Splits text into overlapping chunks
- Creates local Hugging Face embeddings
- Stores embeddings persistently in Chroma
- Retrieves the top five relevant chunks
- Filters weak matches using a distance threshold
- Generates grounded answers using an OpenAI model

## Structure

The pipeline is split into single-responsibility modules, each mirroring one
stage above:

- `config.py` — constants (paths, model names, chunk size/overlap, top-k, distance threshold)
- `loader.py` — PDF loading and text cleaning
- `splitter.py` — chunking
- `embedder.py` — local embedding model
- `vector_store.py` — Chroma storage and dedup
- `retriever.py` — similarity search and threshold filtering
- `generator.py` — prompt building and OpenAI answer generation
- `rag_pipeline.py` — orchestrates the full flow end-to-end
- `app.py` — thin entry point that calls `rag_pipeline.run()`