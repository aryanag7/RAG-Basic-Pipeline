import re
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# Common English words dropped before BM25 scoring. Without this, a query
# like "how do I bake a croissant" still gets a spurious positive BM25 score
# from stopword overlap with unrelated chunks, letting the BM25-only pass-
# through rule (see retriever.py's _passes) let a chunk through even when
# nothing in the document is actually relevant to the query.
# Written as they appear AFTER _tokenize's \w+ split, which breaks
# contractions on the apostrophe (e.g. "don't" -> "don", "t") - so this
# includes the pre-apostrophe stems (e.g. "don", "isn", "shouldn") and the
# leftover fragments ("t", "s", "d", "m", "re", "ve", "ll") rather than the
# whole contractions themselves.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren as at be
    because been before being below between both but by can cannot could
    couldn did didn do does doesn doing don down during each few for
    from further had hadn has hasn have haven having he her here
    hers herself him himself his how i if in into is isn it its itself
    let me more most mustn my myself no nor not of off on once only or
    other ought our ours ourselves out over own same shan she should
    shouldn so some such than that the their theirs them themselves
    then there these they this those through to too under until up
    very was wasn we weren what when where which while who
    whom why will with won would wouldn you your yours yourself yourselves
    t s d m re ve ll
    """.split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer with common English stopwords removed."""

    tokens = re.findall(r"\w+", text.lower())
    return [token for token in tokens if token not in _STOPWORDS]


@dataclass
class BM25Index:
    bm25: BM25Okapi | None
    documents: list[Document]
    ids: list[str]


def build_bm25_index(vector_store: Chroma) -> BM25Index:
    """Build a BM25 index from the current live contents of the vector store."""

    stored = vector_store.get(include=["documents", "metadatas"])

    ids = stored["ids"]
    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(stored["documents"], stored["metadatas"])
    ]

    if not documents:
        return BM25Index(bm25=None, documents=[], ids=[])

    tokenized_corpus = [_tokenize(document.page_content) for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return BM25Index(bm25=bm25, documents=documents, ids=ids)


def search_bm25(
    index: BM25Index,
    query: str,
    k: int,
) -> list[tuple[Document, str, float]]:
    """Return up to k (document, chunk_id, score) tuples, best first."""

    if index.bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    scores = (float(score) for score in index.bm25.get_scores(tokenized_query))

    scored = [
        (document, chunk_id, score)
        for document, chunk_id, score in zip(index.documents, index.ids, scores)
        if score > 0
    ]
    scored.sort(key=lambda item: item[2], reverse=True)

    return scored[:k]
