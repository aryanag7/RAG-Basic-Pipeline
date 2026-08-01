# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SourceLens is a general-purpose Retrieval-Augmented Generation (RAG) application: upload PDFs,
search across your document library, and get clear answers grounded in the original sources. There
is no bundled document and no hardcoded PDF path — the knowledge base is built entirely from PDFs
a user uploads through the Streamlit UI (`streamlit_app.py`), which is the sole front end. The
pipeline — PDF loading, cleaning, chunking, embedding, vector storage, retrieval, and answer
generation — is split into single-responsibility modules (see Architecture below), and
`streamlit_app.py` calls those pipeline functions directly; there is no separate orchestrator or
CLI entry point.

## Setup and running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires an `OPENAI_API_KEY` in a `.env` file at the repo root (loaded via `python-dotenv`).

Run the app:

```bash
streamlit run streamlit_app.py
```

This opens a browser page with an upload widget in the sidebar; the vector store, embedding model,
and LLM are built once per process (`st.cache_resource`) and reused across questions.

There is no test suite, linter, or build step configured in this repo.

## Architecture

There is no `run()`/pipeline orchestrator — `streamlit_app.py` calls each stage's functions
directly, both eagerly at startup (to open the vector store) and per-interaction (upload, ask).

1. **Load** (`loader.py`) — `load_pdf_from_bytes(data, filename)` parses an in-memory uploaded PDF
   via `langchain_community`'s `Blob` + `PyPDFParser` (no temp file) into LangChain `Document`
   objects (one per page). `clean_documents()` / `clean_text()` normalises whitespace and rejoins
   small lowercase fragment lines that PyPDF sometimes splits mid-sentence (`is_list_item`,
   `looks_like_heading` are heuristics used to decide what *not* to rejoin). `load_pdf()` (loading
   a PDF from a fixed path via `PyPDFLoader`) and the `PDF_PATH` constant it depended on
   (`config.py`) are commented out, not deleted — they're unused now that ingestion is upload-only,
   kept only as reference for how a path-based loader would look.
2. **Split** (`splitter.py`) — `split_documents()` uses `RecursiveCharacterTextSplitter`
   (`CHUNK_SIZE`/`CHUNK_OVERLAP` from `config.py`) to turn page-level Documents into smaller chunk
   Documents. `RecursiveCharacterTextSplitter.split_documents()` deep-copies each parent
   `Document.metadata` into every resulting chunk (only adding `start_index` on top) — this is what
   lets `streamlit_app.py` tag a `filename`/`file_size` key on the page-level Documents before
   splitting and have it survive into every stored chunk, see the Streamlit section below.
3. **Embed** (`embedder.py`) — `create_embedding_model()` loads a local Hugging Face model
   (`EMBEDDING_MODEL_NAME`) with normalized embeddings. Embeddings are computed locally; no API
   call.
4. **Store** (`vector_store.py`) — `create_or_load_vector_store(embedding_model)` opens a
   persistent Chroma collection at `vector_store/` (`COLLECTION_NAME`, cosine space) and returns it
   — it no longer takes a `chunks` argument or adds anything itself, since there's no fixed document
   to seed at startup; adding chunks is the caller's job via `add_new_chunks()`. Each chunk gets a
   deterministic sha256 ID (`create_chunk_id`, based on source + page + start_index + content) so
   uploading the same PDF twice — even across process restarts — only adds genuinely new chunks
   instead of duplicating the store. `add_new_chunks(chunks, vector_store)` is factored out
   separately so callers that already hold an open `Chroma` instance (the Streamlit uploader) can
   add to it directly, without constructing a second client.
   - `list_indexed_documents(vector_store)` groups the store's chunks by `metadata['source']` (via
     `vector_store.get(include=["metadatas"])`, since Chroma has no native "distinct values" query
     — the same client-side-grouping approach `bm25_index.py` already uses per-query, and cheaper
     since it only fetches metadata) and returns one summary row per document (`source`,
     `filename`, `size`, `chunk_count`). This is the only source of truth the UI uses to render the
     document list — see the "durable document identity" note below.
   - `delete_source(source, vector_store)` removes every chunk belonging to one document:
     `Chroma.delete()` only accepts `ids` (no `where=` shortcut in this version), so it first
     resolves matching ids via `vector_store.get(where={"source": source})`, then calls
     `vector_store.delete(ids=ids)`. Deletion mutates the shared `Chroma` object in place — its
     identity doesn't change, so no `st.cache_resource` invalidation is needed, just an
     `st.rerun()` so the UI recomputes from the now-smaller store.
5. **Retrieve** (`retriever.py`) — `retrieve_relevant_chunks()` is a hybrid search: an embedding
   channel (Chroma `similarity_search_with_score`, widened to `EMBED_CANDIDATE_K` candidates) and
   a keyword channel (BM25, widened to `BM25_TOP_K` candidates) are fused into one ranked list via
   Reciprocal Rank Fusion (`fusion.py`, constant `RRF_K`), then filtered, reranked with a
   cross-encoder, and truncated to `TOP_K`.
   - `bm25_index.py`'s `build_bm25_index()` rebuilds a `rank_bm25.BM25Okapi` index **from scratch
     on every call**, reading whatever's currently in the vector store via `vector_store.get()` —
     it is not cached. This is deliberate: the Streamlit uploader (see below) adds chunks to the
     already-cached `vector_store` object at arbitrary times after `load_vector_store()` was
     cached, and the new Delete button removes chunks from it too, so rebuilding BM25 fresh from
     live vector-store contents on every query keeps it automatically in sync with both uploads and
     deletes, with no cache-invalidation logic needed. The corpus is small (typically a few hundred
     chunks), so rebuilding per query is cheap. `_tokenize()` lowercases and regex-splits on `\w+`,
     then drops a hardcoded list of common English stopwords written to match tokens *after* that
     split (e.g. `"don't"` becomes `"don"`/`"t"`, so both fragments are listed, not the whole
     contraction) — without this, a query like "how do I bake a croissant" gets a spurious positive
     BM25 score from stopword overlap with unrelated chunks, which would otherwise let it survive
     the gating step below. `search_bm25()` drops any chunk scoring `<= 0` (no real term overlap).
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
   - If nothing survives gating, the OpenAI API is never called — the UI shows a no-match message
     instead of guessing.
   - Reranking (`retriever.py`'s `_rerank()`, model in `reranker.py`): the pool that survives
     gating is scored by a local `sentence-transformers` cross-encoder
     (`RERANKER_MODEL_NAME`, `cross-encoder/ms-marco-MiniLM-L-6-v2`), which jointly encodes each
     `(query, chunk text)` pair rather than comparing independently-computed vectors — normally a
     stronger relevance signal than RRF rank, which only reflects each channel's own ordering. The
     gated pool (not just the top `TOP_K`) is reranked so a chunk that barely survived gating but is
     actually the best semantic match can still rise to the top; only after reranking is the list
     truncated to `TOP_K`. The float returned alongside each `Document` from
     `retrieve_relevant_chunks()` is this cross-encoder score, not the RRF score or an embedding
     distance. Like `create_embedding_model()`/`create_llm()`, `create_reranker()` is wrapped in its
     own `st.cache_resource`-decorated `load_reranker()` in `streamlit_app.py`, loaded lazily on
     the first "Ask" click (not eagerly at module load) and reused across all subsequent queries in
     the process.
   - Every call unconditionally prints each fused candidate's rank, RRF score, per-channel
     distance/score and rank (when present), PASS/DROP verdict, page, source, and a content
     preview to the console — this is permanent, always-on debug logging (not gated behind a
     flag), the main way to diagnose why a query got zero/weak matches or which channel(s)
     contributed a given hit. A second, separate debug block prints the post-rerank order and each
     surviving chunk's `rerank_score`, so it's possible to see exactly how reranking changed the
     order RRF/gating produced.
6. **Generate** (`generator.py`) — `create_llm()` / `generate_answer()` sends the filtered chunks
   plus the question to `ChatOpenAI` (`OPENAI_MODEL_NAME`). The system prompt constrains the
   model to answer only from the provided context and to say so explicitly when the context is
   insufficient.

All tunables live as constants in `config.py`: `EMBEDDING_MODEL_NAME`, `RERANKER_MODEL_NAME`,
`VECTOR_STORE_PATH`, `COLLECTION_NAME`, `TOP_K`, `MAX_DISTANCE`, `EMBED_CANDIDATE_K`, `BM25_TOP_K`,
`RRF_K`, `OPENAI_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP` (`PDF_PATH` is commented out, unused — see the
Load stage above). `EMBED_CANDIDATE_K` and `BM25_TOP_K` (both `20`) widen each channel's candidate
pool before fusion, so RRF and the reranker have enough material to work with beyond just `TOP_K`;
`RRF_K` (`60`) is the standard Reciprocal Rank Fusion constant from Cormack/Clarke/Buettcher 2009.
`config.py` also calls `load_dotenv()`, so it must be imported before anything that reads env
vars — every other module already imports from it, which preserves that ordering.

Debug-only helpers not on the main path: `inspect_documents()` (`loader.py`),
`inspect_one_embedding()` (`embedder.py`), `display_retrieved_chunks()` (`retriever.py` — a
fuller, post-filter-only dump printing each result's fused RRF score, not a raw distance; distinct
from `retrieve_relevant_chunks()`'s own always-on candidate logging described above).

`vector_store/` is a persisted Chroma DB (sqlite + HNSW index files) and is gitignored, along
with `.env`, `.venv/`, and `data/*.pdf`. It starts out absent/empty on a fresh checkout — there is
no bundled document seeded into it — and is only populated by uploads through the Streamlit UI;
don't assume it has any content when reasoning about a fresh checkout.

### Streamlit front end (`streamlit_app.py`)

The sole front end — there is no CLI or pipeline-orchestrator module. Imports the pipeline
building-block functions (`loader.py`, `splitter.py`, `embedder.py`, `vector_store.py`,
`retriever.py`, `generator.py`) directly. Two `st.cache_resource`-wrapped functions open the vector
store and build the LLM once per process, so repeated "Ask" clicks don't reload either;
`load_vector_store()` just opens/creates the Chroma collection (`create_or_load_vector_store`) —
there's no PDF to load, clean, or split at startup anymore. `load_vector_store()` and
`st.session_state["processed_uploads"]` are resolved once near the top of the script, before the
sidebar renders, because the stat cards and documents table need real numbers on first page view.
The "Ask" button mirrors the "no LLM call on zero relevant chunks" rule: the LLM is only
constructed/called when `retrieve_relevant_chunks()` returns non-empty results; otherwise the UI
shows a no-match message. The module also silences the `streamlit.watcher.local_sources_watcher`
logger: Streamlit's file watcher introspects every imported module's `__path__`, and
`transformers` (via `langchain-huggingface`) lazily imports optional vision submodules that need
`torchvision` (not installed, not needed) — the resulting `ModuleNotFoundError` is harmless but
noisy, so it's suppressed rather than adding an unused dependency.

**Durable document identity (`filename`/`file_size` metadata)** — `source` metadata for an upload
is `f"upload:{sha256(file_bytes).hexdigest()}"`, a pure content hash with no human-readable name
(deliberate, so re-uploading byte-identical content under a different filename still dedups). The
original filename therefore has to be captured separately for display, and it has to survive
process restarts and new browser sessions to satisfy persistence — so it's written into Chroma
itself, not just session state: right after `load_pdf_from_bytes()`, the sidebar loop sets
`doc.metadata["filename"] = uploaded_file.name` and `doc.metadata["file_size"] = len(data)` on
every page-level `Document`, *before* `clean_documents()`/`split_documents()`, so it deep-copies
into every resulting chunk (see the Split stage above) and ends up on every record Chroma stores.
`list_indexed_documents()` reads it back out to build the document list on every script run — this
is what makes the table correct after a browser reload or process restart, when
`st.session_state["processed_uploads"]` (session-scoped) would otherwise show nothing even though
Chroma still has the documents.

**Theme (`.streamlit/config.toml`, `ui_theme.py`)** — a "warm editorial" palette (cream
background, ink-navy text, terracotta primary/teal secondary accents, `Fraunces`/`Karla` Google
Fonts) is set entirely through Streamlit 1.60's native `[theme]`/`[theme.sidebar]` config tokens —
deliberately not raw CSS-selector overrides, since Streamlit's internal DOM class names aren't a
stable target across versions. `st.info`/`warning`/`error`/`success` automatically pick up the
theme's semantic colors (`greenColor`/`redColor`/`orangeColor`/`blueColor`/`grayColor`) with no
extra code. `ui_theme.py` covers only what `config.toml` can't express: `CUSTOM_CSS` (injected
once via `st.html()`) holds the pipeline-animation keyframes, one `key=`-scoped rule for the
answer card's accent border, and one for the centered app-title header (`.st-key-<key>` is the
class Streamlit generates for any widget/container given a `key=` — the stable, documented
targeting mechanism this codebase uses instead of internal/unstable selectors); the header rule
also hides `st.title`'s hover anchor-link icon (`[data-testid="stHeaderActionElements"]`, scoped
under `.st-key-app_header` specifically, not globally) since it looks misplaced once the title
block is centered. `render_pipeline_frame(stage)` renders one frame of a 4-node HTML diagram
(Question → Search+fuse → Generate answer → Answer); `format_file_size()` is a small table
formatting helper. No new pip dependency — zero new packages beyond what's already listed in
`requirements.txt`, consistent with this project's preference for built-in-first (see the BM25
stopword list above, hardcoded rather than pulling in NLTK).

**Loading-state pipeline animation** — replaces the previous plain `st.spinner("Thinking...")`.
A single `st.empty()` placeholder is updated with a different animated frame immediately before
each real per-query call: `render_pipeline_frame("retrieve")` before `retrieve_relevant_chunks()`,
then `render_pipeline_frame("generate")` before `generate_answer()` (only reached when `results`
is non-empty — matching the "no LLM call on zero relevant chunks" rule, so the diagram never
implies a generation step that didn't happen), then `render_pipeline_frame("complete")` held for
`time.sleep(0.4)` before the placeholder clears. There is no JavaScript and no client-side timer —
"real progress" comes entirely from which frame the Python code renders and when, synchronized to
the two actual pipeline calls.

**Stat cards + documents table** — three `st.columns` cards (`st.metric` for documents-indexed and
chunks-in-index counts, a `st.badge` callout for "Hybrid search" surfacing the BM25+RRF upgrade
from the Retrieve stage above) sit above a document list built from `list_indexed_documents()`,
called once per script run and reused for both the stat cards and the table so the store is only
queried once. When it returns nothing, a caption ("No documents indexed yet. Upload a PDF to get
started.") is shown instead of an empty table. Otherwise it's rendered as a manual per-row
`st.columns([...])` grid (Document / Size / Chunks / Delete), not `st.table` — `st.table` can't
embed interactive widgets, and each row needs a working Delete button
(`st.button("🗑️", key=f"delete_{source}")`). Clicking it calls `delete_source(source,
vector_store)`, then pops the matching `st.session_state["processed_uploads"]` entry
(`processed.pop(source.removeprefix("upload:"), None)`) — required for correctness, not just
cleanup: that dict also gates the upload loop's same-session skip check, so without popping it, a
user who deletes a document and then re-uploads the identical bytes in the same session would
silently no-op and the document would never come back — then calls `st.rerun()` so the stat cards,
`doc_count`, and table all recompute immediately from the now-smaller store (they're all derived
once at module top-level per script run, not reactively).

**Zero-document guard** — `doc_count = len(documents)` (from the same `list_indexed_documents()`
call above); the Ask button is disabled (`st.button("Ask", type="primary", disabled=(doc_count ==
0))`) and a "Upload a PDF to get started." caption is shown next to it, and in the sidebar near the
uploader, whenever `doc_count == 0`. A disabled `st.button()` never returns `True`, so the existing
`if ask_clicked: ...` branch structure needs no other changes to stay safe with zero documents
indexed.

**PDF upload (sidebar)** — `st.file_uploader(..., accept_multiple_files=True)` lets the user build
up the knowledge base entirely from uploads; all uploaded content lands in the *same* Chroma
collection, so `retrieve_relevant_chunks()`/`generate_answer()` need zero changes to draw on
multiple documents at once — no per-source filtering or merge logic exists anywhere. Processing
pipeline per uploaded file: `loader.load_pdf_from_bytes(data, filename)` parses the in-memory bytes
directly via `langchain_community`'s `Blob` + `PyPDFParser` (the same primitives `PyPDFLoader`
used) — no temp file, and full control over what `metadata['source']` gets set to, which matters
for dedup (see "Durable document identity" above for how the original filename is captured
separately). After `clean_documents()`/`split_documents()`, chunks are added via
`vector_store.add_new_chunks(chunks, vector_store=load_vector_store())`, reusing the same sha256
ID-dedup logic `create_or_load_vector_store()`'s callers rely on elsewhere — no separate dedup
mechanism. `st.session_state["processed_uploads"]` (keyed by content hash) is a **same-session
performance guard and status-message cache**, avoiding re-embedding a file already handled this run
(Streamlit reruns the whole script on every interaction, and the uploader keeps returning the same
files across reruns); the durable "don't add it again" guarantee — surviving process restarts —
comes entirely from `add_new_chunks()`'s persisted ID check, and the durable "what documents
exist" source of truth is `list_indexed_documents()` reading Chroma directly, not this dict. Per-file
outcomes (`added` / `duplicate` / `empty` — zero extractable text, e.g. a scanned PDF — / `error` —
parse failures like encrypted or corrupted PDFs) are all recorded in that same session-state dict,
*including* errors, so a deterministically-failing file doesn't get re-parsed and re-fail on every
subsequent rerun; these render as sidebar banners (`st.success`/`info`/`warning`/`error`) only,
separate from the persistent documents table above.

The upload loop tracks a `newly_processed` flag across the `for uploaded_file in uploaded_files`
loop and, if any file wasn't already in `processed` (i.e. genuinely new work happened this pass),
calls `st.rerun()` once after the loop — the same pattern the delete button already used. This
matters because `documents`/`doc_count`/`total_chunks` (and the stat cards/table built from them)
are computed near the top of the script, *before* this sidebar block runs; without the explicit
rerun, a single script pass would add the chunks to Chroma but still render the stat cards/table
from the pre-upload snapshot taken earlier in that same pass, so the counts looked stale until
something else (like a manual browser reload) triggered a fresh run. The rerun is safe from
looping: on the next pass the same files are still in the uploader's returned list, but their
content hashes are now already in `processed`, so the loop's `if content_hash in processed:
continue` skips them without re-triggering another rerun. Multiple files uploaded together still
only cause one rerun, since the flag is checked once after the whole loop, not per file.

Besides the app's own gitignored paths above, `extract_pdfs_rag/` (a local RAG benchmark
dataset/notebook, unrelated to the pipeline itself and never read by the app — see the next
section) is partially gitignored: the eval script, the benchmark-generation notebook, and the
benchmark metadata CSVs are tracked, but the large/regeneratable pieces are not —
`text_only_rag_benchmark_4x4_unique/pdfs/` (the actual PDF binaries), the two source zip archives,
`eval_vector_store/` and `eval_runs/` (regenerated by every eval run), and the scratch
`All_PDFs.csv`/`Eval_data.csv` files. `.claude/` (machine-local Claude Code settings, e.g.
permission grants tied to one developer's file paths, not project configuration) is also ignored.

### RAG evaluation script (`extract_pdfs_rag/eval.py`)

A standalone script, run manually (`python extract_pdfs_rag/eval.py`), that measures retrieval and
generation quality separately using [DeepEval](https://github.com/confident-ai/deepeval) —
independent of, and without modifying, the app's own pipeline code. The script itself
(`eval.py`), the notebook that built the benchmark
(`create_text_only_rag_benchmark_4x4_unique.ipynb`), the benchmark metadata CSVs
(`evaluation_queries.csv`, `relevant_documents.csv`, `distractor_documents.csv`), and the
cross-run summary (`eval_results.csv`) are tracked in git; the underlying PDF binaries,
per-run vector store, and per-run detail CSVs are not (see the gitignore note above) — so a fresh
checkout has the script and past results but needs the PDFs re-extracted before a run can index
anything.

- **Dataset**: the 80 curated questions in
  `text_only_rag_benchmark_4x4_unique/metadata/evaluation_queries.csv` (query, gold answer, source
  `pdf_filename`, extractive/abstractive `type`), built by
  `create_text_only_rag_benchmark_4x4_unique.ipynb` from 10 "relevant" arXiv PDFs + 5 "distractor"
  PDFs already extracted under `text_only_rag_benchmark_4x4_unique/pdfs/`.
- **Isolation**: indexes those 15 PDFs into `extract_pdfs_rag/eval_vector_store/` (a separate Chroma
  collection, `eval_chunks`) — never the app's production `vector_store/` — rebuilt from scratch on
  every run so the index always reflects `config.py`'s *current* chunking/embedding settings, which
  is what makes it valid to compare runs after changing those constants.
- **Pipeline reuse**: calls the same `loader`/`splitter`/`vector_store`/`reranker`/`retriever`/
  `generator` functions the app uses (`load_pdf_from_bytes`, `clean_documents`, `split_documents`,
  `add_new_chunks`, `create_reranker`, `retrieve_relevant_chunks`, `generate_answer`) — no pipeline
  code was changed to support evaluation. `create_reranker()` is called once per run, the same
  "build once, reuse across all queries" pattern as `create_llm()`, just without
  `st.cache_resource` since there's no long-lived process to cache across. Mirrors the app's "no
  LLM call on empty retrieval" rule per question.
- **Metrics**: retrieval is scored with DeepEval's `ContextualPrecisionMetric`,
  `ContextualRecallMetric`, and `ContextualRelevancyMetric`, plus a ground-truth `hit_rate` (does the
  retrieved set include the query's actual source PDF, checked via chunk `metadata["filename"]`);
  generation is scored with `AnswerRelevancyMetric` and `FaithfulnessMetric`. No `model=` override is
  passed to any metric, so DeepEval's own default judge model is used, and whichever model that
  resolves to is recorded in the output for reproducibility. Per-question retrieval/generation/total
  latency is also measured (`time.perf_counter()` around the real `retrieve_relevant_chunks()` /
  `generate_answer()` calls) and averaged into the summary.
- **Outputs**: one summary row appended per run to `extract_pdfs_rag/eval_results.csv` (a config
  snapshot — chunk size/overlap, embedding model, generation LLM, retrieval tunables — plus each
  metric's mean/pass-rate and the latency means), so different `config.py` settings can be compared
  across runs; and a per-question detail CSV per run under `extract_pdfs_rag/eval_runs/`.
- CLI flags: `--label NAME` (tags the summary row for later comparison) and `--sample-size N`
  (evaluate only the first N questions, for a cheap smoke test before committing to a full run).
