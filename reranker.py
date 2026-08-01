from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME


def create_reranker() -> CrossEncoder:
    """Load the local cross-encoder reranking model."""

    return CrossEncoder(RERANKER_MODEL_NAME)
