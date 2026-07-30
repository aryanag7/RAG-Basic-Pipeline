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
- Retrieves the top five relevant chunks
- Filters weak matches using a distance threshold
- Generates grounded answers using an OpenAI model

If no chunk clears the distance threshold, the OpenAI API is never called — the app says it
couldn't find relevant information instead of guessing.

## Structure

The pipeline is split into single-responsibility modules, each mirroring one stage above:

- `config.py` — constants (paths, model names, chunk size/overlap, top-k, distance threshold)
- `loader.py` — PDF loading and text cleaning
- `splitter.py` — chunking
- `embedder.py` — local embedding model
- `vector_store.py` — Chroma storage and dedup
- `retriever.py` — similarity search and threshold filtering
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

The sidebar lets you upload additional PDFs — each one is loaded, cleaned, chunked, embedded, and
added to the same knowledge base as the bundled PDF, so answers can draw on both. Uploading a PDF
that's already indexed (by content, not filename) is a no-op — it's detected and skipped rather
than duplicated. Per-file status (chunks added, already indexed, no extractable text, or a parse
error) shows in the sidebar as each upload is processed.

## Notes

- There is no test suite, linter, or build step configured in this repo.
- Every retrieval logs each candidate chunk's distance and pass/drop verdict against the
  threshold to the console (CLI terminal or the Streamlit server's terminal) — useful for
  debugging why a question returned no answer.
