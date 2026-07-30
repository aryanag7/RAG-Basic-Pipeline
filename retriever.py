from langchain_chroma import Chroma
from langchain_core.documents import Document

from bm25_index import build_bm25_index, search_bm25
from config import BM25_TOP_K, EMBED_CANDIDATE_K, MAX_DISTANCE, RRF_K, TOP_K
from fusion import FusedResult, reciprocal_rank_fusion
from vector_store import create_chunk_id


def _passes(result: FusedResult, max_distance: float) -> bool:
    """Pass if either channel qualifies the candidate on its own.

    A positive BM25 score is enough on its own, whether or not the same
    chunk also happens to appear in the embedding candidate pool - a chunk
    weakly present in the embedding pool (distance over MAX_DISTANCE)
    shouldn't veto an otherwise dominant keyword match.
    """

    if result.embedding_distance is not None and result.embedding_distance <= max_distance:
        return True

    return result.bm25_score is not None


def retrieve_relevant_chunks(
    query: str,
    vector_store: Chroma,
    top_k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
    embed_candidate_k: int = EMBED_CANDIDATE_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """Retrieve relevant chunks via hybrid embedding + BM25 search.

    Runs both retrieval channels over widened candidate pools, fuses them
    with Reciprocal Rank Fusion, filters weak matches, and returns the top
    results. The returned float is the fused RRF score (not a distance).
    """

    embedding_hits = vector_store.similarity_search_with_score(
        query=query,
        k=embed_candidate_k,
    )
    embedding_results = [
        (document, create_chunk_id(document), distance)
        for document, distance in embedding_hits
    ]

    bm25_index = build_bm25_index(vector_store)
    bm25_results = search_bm25(bm25_index, query, k=bm25_top_k)

    fused = reciprocal_rank_fusion(embedding_results, bm25_results, rrf_k=rrf_k)

    print(
        f"\nRetrieval candidates for query: {query!r} "
        f"(embed_candidate_k={embed_candidate_k}, bm25_top_k={bm25_top_k}, "
        f"max_distance={max_distance}, rrf_k={rrf_k})"
    )
    for rank, result in enumerate(fused, start=1):
        verdict = "PASS" if _passes(result, max_distance) else "DROP"
        page = result.document.metadata.get("page_label", "Unknown")
        source = result.document.metadata.get("source", "Unknown")
        preview = result.document.page_content[:80].replace("\n", " ")

        channels = []
        if result.embedding_rank is not None:
            channels.append(f"embed_dist={result.embedding_distance:.4f}(rank{result.embedding_rank})")
        if result.bm25_rank is not None:
            channels.append(f"bm25_score={result.bm25_score:.4f}(rank{result.bm25_rank})")

        print(
            f"  [{verdict}] #{rank} rrf_score={result.rrf_score:.4f} "
            f"{' '.join(channels)} page={page} source={source} | {preview!r}"
        )

    passing_results = [result for result in fused if _passes(result, max_distance)]

    return [
        (result.document, result.rrf_score)
        for result in passing_results[:top_k]
    ]


def display_retrieved_chunks(
    results: list[tuple[Document, float]],
) -> None:
    """Display retrieved chunks and their source information."""

    for rank, (document, score) in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print(f"Result: {rank}")
        print(f"Relevance score (RRF): {score:.4f}")
        print(f"Page: {document.metadata.get('page_label')}")
        print(f"Source: {document.metadata.get('source')}")
        print("\nRetrieved text:")
        print(document.page_content)
