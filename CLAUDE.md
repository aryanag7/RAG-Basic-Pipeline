# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A work-in-progress Retrieval-Augmented Generation (RAG) application that answers system-design
questions from a fixed PDF document (`data/System Design Concepts.pdf`), plus any PDFs a user
uploads through the Streamlit UI. The pipeline — PDF loading, cleaning, chunking, embedding,
vector storage, retrieval, and answer generation — is split into single-responsibility modules
(see Architecture below). It has two front ends: a CLI orchestrated by `rag_pipeline.py` and run
via the thin entry point `app.py` (fixed PDF only), and a Streamlit UI (`streamlit_app.py`) that
calls the same underlying pipeline functions directly and additionally supports uploading more
PDFs into the same knowledge base.

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

Run the Streamlit UI instead:

```bash
streamlit run streamlit_app.py
```

This opens a browser page with a query box and an "Ask" button; the vector store, embedding
model, and LLM are built once per process (`st.cache_resource`) and reused across questions.

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
   collection at `vector_store/` (`COLLECTION_NAME`, cosine space) and delegates the actual
   dedup-and-add work to `add_new_chunks(chunks, vector_store)`. Each chunk gets a deterministic
   sha256 ID (`create_chunk_id`, based on source + page + start_index + content) so re-running the
   pipeline — or adding a Streamlit-uploaded PDF, see below — only adds genuinely new chunks
   instead of duplicating the store. `add_new_chunks()` is factored out separately so callers that
   already hold an open `Chroma` instance (the Streamlit uploader) can add to it directly, without
   constructing a second client.
5. **Retrieve** (`retriever.py`) — `retrieve_relevant_chunks()` is a hybrid search: an embedding
   channel (Chroma `similarity_search_with_score`, widened to `EMBED_CANDIDATE_K` candidates) and
   a keyword channel (BM25, widened to `BM25_TOP_K` candidates) are fused into one ranked list via
   Reciprocal Rank Fusion (`fusion.py`, constant `RRF_K`), then filtered and truncated to `TOP_K`.
   - `bm25_index.py`'s `build_bm25_index()` rebuilds a `rank_bm25.BM25Okapi` index **from scratch
     on every call**, reading whatever's currently in the vector store via `vector_store.get()` —
     it is not cached and not built from the `chunks` list computed earlier in the pipeline. This
     is deliberate: the Streamlit uploader (see below) adds chunks to the already-cached
     `vector_store` object at arbitrary times after `load_vector_store()` was cached, so rebuilding
     BM25 fresh from live vector-store contents on every query keeps it automatically in sync with
     uploads, with no cache-invalidation logic needed. The corpus is small (a few hundred chunks),
     so rebuilding per query is cheap. `_tokenize()` lowercases and regex-splits on `\w+`, then
     drops a hardcoded list of common English stopwords written to match tokens *after* that split
     (e.g. `"don't"` becomes `"don"`/`"t"`, so both fragments are listed, not the whole
     contraction) — without this, a query like "how do I bake a croissant" gets a spurious
     positive BM25 score from stopword overlap with unrelated chunks, which would otherwise let it
     survive the gating step below. `search_bm25()` drops any chunk scoring `<= 0` (no real term
     overlap).
   - The embedding side needs a join key to match against BM25's native chunk ids (which
     `vector_store.get()` returns for free); `similarity_search_with_score()` doesn't return ids,
     so `retrieve_relevant_chunks()` recomputes them by calling the existing `create_chunk_id()`
     from `vector_store.py` on each hit, reusing the same deterministic sha256 logic rather than
     adding a second ID scheme.
   - `fusion.py`'s `reciprocal_rank_fusion()` sums `1 / (RRF_K + rank)` per chunk id across
     whichever list(s) it appears in (1-based rank), returning the full fused list sorted by RRF
     score — un-truncated and un-gated; `retriever.py` applies gating/truncation afterward.
   - Gating (`retriever.py`'s `_passes()`): a candidate passes if **either** channel independently
     qualifies it — embedding distance `<= MAX_DISTANCE`, or a positive BM25 score — regardless of
     whether it also appears in the other channel. This is deliberately OR, not "embedding-pool
     presence vetoes the BM25 pass": an earlier version required embedding distance to pass
     whenever a chunk was present in the embedding pool at all, which caused a real miss in
     testing — a chunk with an overwhelming, unique-term BM25 match (rank 1, score roughly double
     any other seen in testing) got dropped because it also weakly showed up in the embedding pool
     just over `MAX_DISTANCE`. OR logic is a strict superset of what passed before and fixes that
     case without weakening anything else.
   - If nothing survives gating, the OpenAI API is never called (see the `else` branch in
     `rag_pipeline.run()`) — unchanged from before hybrid search.
   - Every call unconditionally prints each fused candidate's rank, RRF score, per-channel
     distance/score and rank (when present), PASS/DROP verdict, page, source, and a content
     preview to the console — this is permanent, always-on debug logging (not gated behind a
     flag), the main way to diagnose why a query got zero/weak matches or which channel(s)
     contributed a given hit.
6. **Generate** (`generator.py`) — `create_llm()` / `generate_answer()` sends the filtered chunks
   plus the question to `ChatOpenAI` (`OPENAI_MODEL_NAME`). The system prompt constrains the
   model to answer only from the provided context and to say so explicitly when the context is
   insufficient.

All tunables live as constants in `config.py`: `PDF_PATH`, `EMBEDDING_MODEL_NAME`,
`VECTOR_STORE_PATH`, `COLLECTION_NAME`, `TOP_K`, `MAX_DISTANCE`, `EMBED_CANDIDATE_K`,
`BM25_TOP_K`, `RRF_K`, `OPENAI_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`. `EMBED_CANDIDATE_K`
and `BM25_TOP_K` (both `20`, 4x `TOP_K`) widen each channel's candidate pool before fusion, so RRF
has enough material to re-rank across channels; `RRF_K` (`60`) is the standard Reciprocal Rank
Fusion constant from Cormack/Clarke/Buettcher 2009. `config.py` also calls `load_dotenv()`, so it
must be imported before anything that reads env vars — every other module already imports from
it, which preserves that ordering.

Debug-only helpers not on the main pipeline path: `inspect_documents()` (`loader.py`),
`inspect_one_embedding()` (`embedder.py`), `display_retrieved_chunks()` (`retriever.py` — a
fuller, post-filter-only dump printing each result's fused RRF score, not a raw distance; distinct
from `retrieve_relevant_chunks()`'s own always-on candidate logging described above).

`vector_store/` is a persisted Chroma DB (sqlite + HNSW index files) and is gitignored, along
with `.env`, `.venv/`, and `data/*.pdf`. Since the PDF itself is gitignored, don't assume it's
present when reasoning about a fresh checkout.

### Streamlit front end (`streamlit_app.py`)

Imports the same building-block functions the stages above use (`loader.py`, `splitter.py`,
`embedder.py`, `vector_store.py`, `retriever.py`, `generator.py`) directly, independent of
`rag_pipeline.run()` — the CLI orchestrator stays the sole caller of `run()`. Two
`st.cache_resource`-wrapped functions build the vector store (load → clean → split → embed →
store, combined into one cached call) and the LLM once per process, so repeated "Ask" clicks
don't reload either. The "Ask" button mirrors `rag_pipeline.run()`'s `if results: ... else: ...`
branch: the LLM is only constructed/called when `retrieve_relevant_chunks()` returns non-empty
results; otherwise the UI shows the same no-match string the CLI prints
(`"I could not find relevant information in the provided document."`, distinct from the
context-insufficient string baked into `generator.py`'s system prompt). The module also silences
the `streamlit.watcher.local_sources_watcher` logger: Streamlit's file watcher introspects every
imported module's `__path__`, and `transformers` (via `langchain-huggingface`) lazily imports
optional vision submodules that need `torchvision` (not installed, not needed) — the resulting
`ModuleNotFoundError` is harmless but noisy, so it's suppressed rather than adding an unused
dependency.

**PDF upload (sidebar)** — `st.file_uploader(..., accept_multiple_files=True)` lets the user add
PDFs beyond the fixed one; uploaded content is merged into the *same* Chroma collection, so
`retrieve_relevant_chunks()`/`generate_answer()` need zero changes to draw on both sources — no
per-source filtering or merge logic exists anywhere. Processing pipeline per uploaded file:
`loader.load_pdf_from_bytes(data, filename)` parses the in-memory bytes directly via
`langchain_community`'s `Blob` + `PyPDFParser` (the same primitives `PyPDFLoader` uses
internally) — no temp file, and full control over what `metadata['source']` gets set to, which
matters for dedup. `source` is set to `f"upload:{sha256(file_bytes).hexdigest()}"` — a **pure
content hash, deliberately excluding the filename** — so re-uploading byte-identical content
under a different filename still produces the same chunk IDs and gets deduped; the filename is
tracked only in `st.session_state` for display. After `clean_documents()`/`split_documents()`,
chunks are added via `vector_store.add_new_chunks(chunks, vector_store=load_vector_store())`,
reusing the exact same sha256 ID-dedup logic `create_or_load_vector_store()` uses for the CLI —
no separate dedup mechanism. `st.session_state["processed_uploads"]` (keyed by content hash) is a
**same-session performance guard only**, avoiding re-embedding a file already handled this run
(Streamlit reruns the whole script on every interaction, and the uploader keeps returning the
same files across reruns); the durable "don't add it again" guarantee — surviving process
restarts — comes entirely from `add_new_chunks()`'s persisted ID check. Per-file outcomes
(`added` / `duplicate` / `empty` — zero extractable text, e.g. a scanned PDF — / `error` — parse
failures like encrypted or corrupted PDFs) are all recorded in that same session-state dict,
*including* errors, so a deterministically-failing file doesn't get re-parsed and re-fail on
every subsequent rerun.
