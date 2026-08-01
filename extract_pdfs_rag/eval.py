"""Evaluate SourceLens's retrieval and generation pipelines with DeepEval.

Indexes the curated benchmark PDFs (text_only_rag_benchmark_4x4_unique/) into an
isolated Chroma store, runs each benchmark query through the real
retrieve_relevant_chunks() / generate_answer() pipeline, scores the results with
DeepEval's contextual (retrieval) and answer-quality (generation) metrics, and
appends one summary row per run to eval_results.csv so different config.py
settings can be compared across runs.
"""

import argparse
import csv
import shutil
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_chroma import Chroma  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402

import config  # noqa: E402
from embedder import create_embedding_model  # noqa: E402
from generator import create_llm, generate_answer  # noqa: E402
from loader import clean_documents, load_pdf_from_bytes  # noqa: E402
from reranker import create_reranker  # noqa: E402
from retriever import retrieve_relevant_chunks  # noqa: E402
from splitter import split_documents  # noqa: E402
from vector_store import add_new_chunks  # noqa: E402

EVAL_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = EVAL_ROOT / "text_only_rag_benchmark_4x4_unique"
QUERIES_CSV = BENCHMARK_DIR / "metadata" / "evaluation_queries.csv"
RELEVANT_PDFS_DIR = BENCHMARK_DIR / "pdfs" / "relevant"
DISTRACTOR_PDFS_DIR = BENCHMARK_DIR / "pdfs" / "distractors"

EVAL_STORE_PATH = EVAL_ROOT / "eval_vector_store"
EVAL_COLLECTION_NAME = "eval_chunks"

RESULTS_CSV = EVAL_ROOT / "eval_results.csv"
RUNS_DIR = EVAL_ROOT / "eval_runs"

RESULTS_FIELDNAMES = [
    "run_id",
    "run_timestamp",
    "label",
    "num_queries_total",
    "num_queries_evaluated",
    "hit_rate",
    "chunk_size",
    "chunk_overlap",
    "embedding_model",
    "generation_llm",
    "top_k",
    "max_distance",
    "embed_candidate_k",
    "bm25_top_k",
    "rrf_k",
    "deepeval_judge_model",
    "contextual_precision_mean",
    "contextual_precision_pass_rate",
    "contextual_recall_mean",
    "contextual_recall_pass_rate",
    "contextual_relevancy_mean",
    "contextual_relevancy_pass_rate",
    "answer_relevancy_mean",
    "answer_relevancy_pass_rate",
    "faithfulness_mean",
    "faithfulness_pass_rate",
    "retrieval_time_mean_sec",
    "generation_time_mean_sec",
    "total_time_mean_sec",
]


def load_queries(sample_size: int | None) -> list[dict]:
    with QUERIES_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if sample_size is not None:
        rows = rows[:sample_size]

    return rows


def build_eval_vector_store() -> Chroma:
    """Index the benchmark's relevant + distractor PDFs into a fresh, isolated store.

    Rebuilt from scratch on every run (not the production vector_store/) so the
    index always reflects config.py's *current* chunking/embedding settings.
    """

    if EVAL_STORE_PATH.exists():
        shutil.rmtree(EVAL_STORE_PATH)

    embedding_model = create_embedding_model()
    vector_store = Chroma(
        collection_name=EVAL_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(EVAL_STORE_PATH),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )

    pdf_paths = sorted(RELEVANT_PDFS_DIR.glob("*.pdf")) + sorted(DISTRACTOR_PDFS_DIR.glob("*.pdf"))

    for pdf_path in pdf_paths:
        data = pdf_path.read_bytes()
        pages = load_pdf_from_bytes(data, filename=f"eval:{pdf_path.name}")

        for page in pages:
            page.metadata["filename"] = pdf_path.name
            page.metadata["file_size"] = len(data)

        cleaned = clean_documents(pages)
        chunks = split_documents(cleaned)
        added = add_new_chunks(chunks, vector_store=vector_store) if chunks else 0
        print(f"Indexed {pdf_path.name}: {len(chunks)} chunks ({added} new)")

    return vector_store


def build_metrics() -> dict:
    """Instantiate DeepEval metrics with no model= override, so DeepEval's own default judge is used."""

    return {
        "contextual_precision": ContextualPrecisionMetric(),
        "contextual_recall": ContextualRecallMetric(),
        "contextual_relevancy": ContextualRelevancyMetric(),
        "answer_relevancy": AnswerRelevancyMetric(),
        "faithfulness": FaithfulnessMetric(),
    }


def mean_or_blank(values: list[float]):
    return statistics.mean(values) if values else ""


def run_evaluation(sample_size: int | None, label: str) -> None:
    queries = load_queries(sample_size)
    print(f"Loaded {len(queries)} queries from {QUERIES_CSV.relative_to(REPO_ROOT)}")

    vector_store = build_eval_vector_store()
    llm = create_llm()
    reranker = create_reranker()
    metrics = build_metrics()
    deepeval_judge_model = str(next(iter(metrics.values())).evaluation_model)

    detail_fieldnames = [
        "query_id",
        "query",
        "type",
        "answer",
        "pdf_filename",
        "generated_answer",
        "retrieved_sources",
        "hit",
        "retrieval_time_sec",
        "generation_time_sec",
        "total_time_sec",
    ]
    for name in metrics:
        detail_fieldnames += [f"{name}_score", f"{name}_reason"]

    metric_scores = {name: [] for name in metrics}
    metric_successes = {name: [] for name in metrics}
    detail_rows = []
    hits = 0
    evaluated = 0
    retrieval_times = []
    generation_times = []
    total_times = []

    for index, row in enumerate(queries, start=1):
        query = row["query"]
        gold_answer = row["answer"]
        gold_pdf_filename = row["pdf_filename"]

        retrieval_start = time.perf_counter()
        results = retrieve_relevant_chunks(query=query, vector_store=vector_store, reranker=reranker)
        retrieval_time = time.perf_counter() - retrieval_start
        retrieval_times.append(retrieval_time)

        retrieved_filenames = [doc.metadata.get("filename") for doc, _ in results]
        hit = gold_pdf_filename in retrieved_filenames
        hits += int(hit)

        detail_row = {
            "query_id": row["query_id"],
            "query": query,
            "type": row.get("type", ""),
            "answer": gold_answer,
            "pdf_filename": gold_pdf_filename,
            "generated_answer": "",
            "retrieved_sources": ";".join(f for f in retrieved_filenames if f),
            "hit": hit,
            "retrieval_time_sec": retrieval_time,
        }

        if not results:
            # Mirrors streamlit_app.py: no LLM call when retrieval returns nothing.
            detail_row["generated_answer"] = "[skipped: empty retrieval]"
            detail_row["generation_time_sec"] = ""
            detail_row["total_time_sec"] = retrieval_time
            total_times.append(retrieval_time)
            for name in metrics:
                detail_row[f"{name}_score"] = ""
                detail_row[f"{name}_reason"] = ""
            detail_rows.append(detail_row)
            print(
                f"[{index}/{len(queries)}] hit={hit} retrieval={retrieval_time:.2f}s "
                "(empty retrieval, generation skipped)"
            )
            continue

        generation_start = time.perf_counter()
        answer = generate_answer(query=query, results=results, llm=llm)
        generation_time = time.perf_counter() - generation_start
        generation_times.append(generation_time)
        total_time = retrieval_time + generation_time
        total_times.append(total_time)

        detail_row["generated_answer"] = answer
        detail_row["generation_time_sec"] = generation_time
        detail_row["total_time_sec"] = total_time
        evaluated += 1

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=gold_answer,
            retrieval_context=[doc.page_content for doc, _ in results],
        )

        for name, metric in metrics.items():
            metric.measure(test_case)
            metric_scores[name].append(metric.score)
            metric_successes[name].append(bool(metric.success))
            detail_row[f"{name}_score"] = metric.score
            detail_row[f"{name}_reason"] = metric.reason

        detail_rows.append(detail_row)
        print(
            f"[{index}/{len(queries)}] hit={hit} retrieval={retrieval_time:.2f}s "
            f"generation={generation_time:.2f}s total={total_time:.2f}s"
        )

    run_id = uuid.uuid4().hex[:12]
    run_timestamp = datetime.now(timezone.utc).isoformat()

    summary_row = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "label": label,
        "num_queries_total": len(queries),
        "num_queries_evaluated": evaluated,
        "hit_rate": hits / len(queries) if queries else "",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "generation_llm": config.OPENAI_MODEL_NAME,
        "top_k": config.TOP_K,
        "max_distance": config.MAX_DISTANCE,
        "embed_candidate_k": config.EMBED_CANDIDATE_K,
        "bm25_top_k": config.BM25_TOP_K,
        "rrf_k": config.RRF_K,
        "deepeval_judge_model": deepeval_judge_model,
    }
    for name in metrics:
        summary_row[f"{name}_mean"] = mean_or_blank(metric_scores[name])
        summary_row[f"{name}_pass_rate"] = mean_or_blank(
            [1.0 if success else 0.0 for success in metric_successes[name]]
        )
    summary_row["retrieval_time_mean_sec"] = mean_or_blank(retrieval_times)
    summary_row["generation_time_mean_sec"] = mean_or_blank(generation_times)
    summary_row["total_time_mean_sec"] = mean_or_blank(total_times)

    # Rewrite (rather than append) so that if RESULTS_FIELDNAMES has grown since
    # earlier runs (e.g. this run added latency columns), the header always
    # matches the current schema and older rows just get blank values for the
    # new columns, instead of silently misaligning the CSV.
    existing_rows = []
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(summary_row)

    RUNS_DIR.mkdir(exist_ok=True)
    details_path = RUNS_DIR / f"{run_id}_details.csv"
    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\n" + "=" * 60)
    print(f"Run {run_id} complete: {evaluated}/{len(queries)} queries evaluated, "
          f"hit_rate={summary_row['hit_rate']}")
    for name in metrics:
        print(f"  {name}: mean={summary_row[f'{name}_mean']} "
              f"pass_rate={summary_row[f'{name}_pass_rate']}")
    if retrieval_times:
        retrieval_part = f"retrieval_mean={summary_row['retrieval_time_mean_sec']:.2f}s"
        if generation_times:
            print(
                f"  latency: {retrieval_part} "
                f"generation_mean={summary_row['generation_time_mean_sec']:.2f}s "
                f"total_mean={summary_row['total_time_mean_sec']:.2f}s"
            )
        else:
            print(f"  latency: {retrieval_part} (no generation ran)")
    print(f"Summary appended to {RESULTS_CSV.relative_to(REPO_ROOT)}")
    print(f"Per-query detail written to {details_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SourceLens's retrieval and generation pipelines with DeepEval."
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional tag for this run, stored in eval_results.csv (e.g. 'baseline').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate only the first N queries (default: all queries in evaluation_queries.csv).",
    )
    args = parser.parse_args()

    run_evaluation(sample_size=args.sample_size, label=args.label)


if __name__ == "__main__":
    main()
