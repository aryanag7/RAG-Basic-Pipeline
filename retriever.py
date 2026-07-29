from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import MAX_DISTANCE, TOP_K


def retrieve_relevant_chunks(
    query: str,
    vector_store: Chroma,
    top_k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
) -> list[tuple[Document, float]]:
    """Retrieve relevant chunks and remove weak matches."""

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    relevant_results = [
        (document, distance)
        for document, distance in results
        if distance <= max_distance
    ]

    return relevant_results


def display_retrieved_chunks(
    results: list[tuple[Document, float]],
) -> None:
    """Display retrieved chunks and their source information."""

    for rank, (document, score) in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print(f"Result: {rank}")
        print(f"Distance score: {score:.4f}")
        print(f"Page: {document.metadata.get('page_label')}")
        print(f"Source: {document.metadata.get('source')}")
        print("\nRetrieved text:")
        print(document.page_content)
