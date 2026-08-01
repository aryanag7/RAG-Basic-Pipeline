from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# Unused now that ingestion is upload-only (see loader.py's commented-out load_pdf()).
# Kept for reference.
# PDF_PATH = Path("data") / "System Design Concepts.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_STORE_PATH = Path("vector_store")
COLLECTION_NAME = "document_chunks"
TOP_K = 7
MAX_DISTANCE = 0.75
EMBED_CANDIDATE_K = 20
BM25_TOP_K = 20
RRF_K = 60
OPENAI_MODEL_NAME = "gpt-5-mini"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 175
