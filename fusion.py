from dataclasses import dataclass

from langchain_core.documents import Document

from config import RRF_K


@dataclass
class FusedResult:
    document: Document
    chunk_id: str
    rrf_score: float
    embedding_distance: float | None
    bm25_score: float | None
    embedding_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    embedding_results: list[tuple[Document, str, float]],
    bm25_results: list[tuple[Document, str, float]],
    rrf_k: int = RRF_K,
) -> list[FusedResult]:
    """Fuse two best-first ranked candidate lists into one, by chunk id."""

    fused: dict[str, FusedResult] = {}

    for rank, (document, chunk_id, distance) in enumerate(embedding_results, start=1):
        fused[chunk_id] = FusedResult(
            document=document,
            chunk_id=chunk_id,
            rrf_score=1 / (rrf_k + rank),
            embedding_distance=distance,
            bm25_score=None,
            embedding_rank=rank,
            bm25_rank=None,
        )

    for rank, (document, chunk_id, score) in enumerate(bm25_results, start=1):
        contribution = 1 / (rrf_k + rank)

        if chunk_id in fused:
            existing = fused[chunk_id]
            existing.rrf_score += contribution
            existing.bm25_score = score
            existing.bm25_rank = rank
        else:
            fused[chunk_id] = FusedResult(
                document=document,
                chunk_id=chunk_id,
                rrf_score=contribution,
                embedding_distance=None,
                bm25_score=score,
                embedding_rank=None,
                bm25_rank=rank,
            )

    return sorted(fused.values(), key=lambda result: result.rrf_score, reverse=True)
