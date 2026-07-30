# System Design RAG

A work-in-progress Retrieval-Augmented Generation (RAG) application that answers system-design
questions grounded in a fixed PDF document — and, through the Streamlit UI, any additional PDFs
you upload. Ask a question through a CLI or a Streamlit web UI and get an answer generated only
from the content of the document(s), with page references.

## How it works

- Loads and cleans a PDF
- Splits text into overlapping chunks
- Creates local Hugging Face embeddings
- Stores embeddings persistently in Chroma
- Retrieves candidates via **hybrid search**: embedding similarity (Chroma) and keyword search
  (BM25), combined with Reciprocal Rank Fusion (RRF)
- Filters weak matches — a chunk survives if either channel independently qualifies it (embedding
  distance under the threshold, or a positive BM25 keyword-overlap score)
- Keeps the top five fused results
- Generates grounded answers using an OpenAI model

If no chunk clears that bar, the OpenAI API is never called — the app says it couldn't find
relevant information instead of guessing.

## Structure

The pipeline is split into single-responsibility modules, each mirroring one stage above:

- `config.py` — constants (paths, model names, chunk size/overlap, top-k, distance threshold)
- `loader.py` — PDF loading and text cleaning
- `splitter.py` — chunking
- `embedder.py` — local embedding model
- `vector_store.py` — Chroma storage and dedup
- `bm25_index.py` — BM25 keyword index, rebuilt from the live vector store on every query
- `fusion.py` — Reciprocal Rank Fusion of the embedding and BM25 result lists
- `retriever.py` — runs both retrieval channels, fuses them, and applies threshold filtering
- `generator.py` — prompt building and OpenAI answer generation
- `rag_pipeline.py` — orchestrates the full flow end-to-end
- `app.py` — thin CLI entry point that calls `rag_pipeline.run()`
- `streamlit_app.py` — Streamlit web UI, calling the same pipeline modules directly

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add an `OPENAI_API_KEY` to a `.env` file at the repo root, and place the source PDF at
`data/System Design Concepts.pdf`.

## Usage

### CLI

```bash
python app.py
```

Builds/updates the Chroma vector store from the PDF, then prompts for a question on stdin and
prints a grounded answer.

### Streamlit web UI

```bash
streamlit run streamlit_app.py
```

Opens a browser page with a query box and an "Ask" button. The embedding model, vector store,
and LLM are loaded once per process and reused across questions, so only the first question pays
the index-build cost.

The main page shows two stat cards (documents indexed, chunks in the index) plus a table of every
indexed document — the bundled PDF and any uploads, each with a size and a status badge — so you
can see exactly what's in the knowledge base at a glance.

The sidebar lets you upload additional PDFs — each one is loaded, cleaned, chunked, embedded, and
added to the same knowledge base as the bundled PDF, so answers can draw on both. Uploading a PDF
that's already indexed (by content, not filename) is a no-op — it's detected and skipped rather
than duplicated. Per-file status (chunks added, already indexed, no extractable text, or a parse
error) shows in the sidebar as each upload is processed, and in the documents table above.

Asking a question replaces the loading spinner with a small animated diagram (Question → Search +
fuse → Generate answer → Answer) that advances in step with what's actually happening — the
"Search + fuse" node lights up while the hybrid retriever runs, and "Generate answer" only lights
up once the LLM call actually starts.

## Notes

- There is no test suite, linter, or build step configured in this repo.
- Every retrieval logs each fused candidate's rank, RRF score, per-channel distance/score, and
  pass/drop verdict to the console (CLI terminal or the Streamlit server's terminal) — useful for
  debugging why a question returned no answer or which channel (embedding, BM25, or both)
  surfaced a given chunk.
