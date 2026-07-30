from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PDF_PATH = Path("data") / "System Design Concepts.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = Path("vector_store")
COLLECTION_NAME = "system_design_concepts"
TOP_K = 5
MAX_DISTANCE = 0.75
EMBED_CANDIDATE_K = 20
BM25_TOP_K = 20
RRF_K = 60
OPENAI_MODEL_NAME = "gpt-5-mini"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
