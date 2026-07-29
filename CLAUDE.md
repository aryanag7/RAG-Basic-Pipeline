# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A work-in-progress Retrieval-Augmented Generation (RAG) application that answers system-design
questions from a single PDF document (`data/System Design Concepts.pdf`). The pipeline — PDF
loading, cleaning, chunking, embedding, vector storage, retrieval, and answer generation — is
split into single-responsibility modules (see Architecture below), orchestrated by
`rag_pipeline.py` and run via the thin entry point `app.py`.

## Setup and running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires an `OPENAI_API_KEY` in a `.env` file at the repo root (loaded via `python-dotenv`).

Run the app (interactive CLI):

```bash
python app.py
```

This builds/updates the Chroma vector store from the PDF, then prompts for a question on stdin
and prints a grounded answer.

There is no test suite, linter, or build step configured in this repo.

## Architecture

`rag_pipeline.run()` orchestrates the stages below in order, top-to-bottom, and is called from
`app.py`'s `if __name__ == "__main__"` block — that's the only place `run()` is invoked:

1. **Load** (`loader.py`) — `load_pdf()` reads `data/System Design Concepts.pdf` via
   `PyPDFLoader` into LangChain `Document` objects (one per page). `clean_documents()` /
   `clean_text()` normalises whitespace and rejoins small lowercase fragment lines that PyPDF
   sometimes splits mid-sentence (`is_list_item`, `looks_like_heading` are heuristics used to
   decide what *not* to rejoin).
2. **Split** (`splitter.py`) — `split_documents()` uses `RecursiveCharacterTextSplitter`
   (`CHUNK_SIZE`/`CHUNK_OVERLAP` from `config.py`) to turn page-level Documents into smaller
   chunk Documents.
3. **Embed** (`embedder.py`) — `create_embedding_model()` loads a local Hugging Face model
   (`EMBEDDING_MODEL_NAME`) with normalized embeddings. Embeddings are computed locally; no API
   call.
4. **Store** (`vector_store.py`) — `create_or_load_vector_store()` opens a persistent Chroma
   collection at `vector_store/` (`COLLECTION_NAME`, cosine space). Each chunk gets a
   deterministic sha256 ID (`create_chunk_id`, based on source + page + start_index + content) so
   re-running the pipeline only adds genuinely new chunks instead of duplicating the store.
5. **Retrieve** (`retriever.py`) — `retrieve_relevant_chunks()` does a similarity search
   (`TOP_K`) and drops any result with distance above `MAX_DISTANCE`. If nothing survives the
   threshold, the OpenAI API is never called (see the `else` branch in `rag_pipeline.run()`).
6. **Generate** (`generator.py`) — `create_llm()` / `generate_answer()` sends the filtered chunks
   plus the question to `ChatOpenAI` (`OPENAI_MODEL_NAME`). The system prompt constrains the
   model to answer only from the provided context and to say so explicitly when the context is
   insufficient.

All tunables live as constants in `config.py`: `PDF_PATH`, `EMBEDDING_MODEL_NAME`,
`VECTOR_STORE_PATH`, `COLLECTION_NAME`, `TOP_K`, `MAX_DISTANCE`, `OPENAI_MODEL_NAME`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`. `config.py` also calls `load_dotenv()`, so it must be imported
before anything that reads env vars — every other module already imports from it, which
preserves that ordering.

Debug-only helpers not on the main pipeline path: `inspect_documents()` (`loader.py`),
`inspect_one_embedding()` (`embedder.py`), `display_retrieved_chunks()` (`retriever.py`).

`vector_store/` is a persisted Chroma DB (sqlite + HNSW index files) and is gitignored, along
with `.env`, `.venv/`, and `data/*.pdf`. Since the PDF itself is gitignored, don't assume it's
present when reasoning about a fresh checkout.
