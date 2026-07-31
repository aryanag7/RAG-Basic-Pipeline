from hashlib import sha256

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import COLLECTION_NAME, VECTOR_STORE_PATH


def create_chunk_id(chunk: Document) -> str:
    """Create a repeatable unique ID for a chunk."""

    identity = (
        f"{chunk.metadata.get('source')}|"
        f"{chunk.metadata.get('page')}|"
        f"{chunk.metadata.get('start_index')}|"
        f"{chunk.page_content}"
    )

    return sha256(identity.encode("utf-8")).hexdigest()

def add_new_chunks(chunks: list[Document], vector_store: Chroma) -> int:
    """Add only chunks not already stored (by deterministic sha256 ID). Returns count added."""

    chunk_ids = [create_chunk_id(chunk) for chunk in chunks]

    stored_data = vector_store.get()
    existing_ids = set(stored_data["ids"])

    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

    return len(new_chunks)


def create_or_load_vector_store(embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Open the persistent Chroma collection, creating it if it doesn't exist yet."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_STORE_PATH),
        collection_configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )


def list_indexed_documents(vector_store: Chroma) -> list[dict]:
    """Group stored chunks by source, returning one summary row per document."""

    stored_data = vector_store.get(include=["metadatas"])

    documents: dict[str, dict] = {}

    for metadata in stored_data["metadatas"]:
        source = metadata.get("source")
        document = documents.setdefault(
            source,
            {
                "source": source,
                "filename": metadata.get("filename", source),
                "size": metadata.get("file_size", 0),
                "chunk_count": 0,
            },
        )
        document["chunk_count"] += 1

    return sorted(documents.values(), key=lambda document: document["filename"])


def delete_source(source: str, vector_store: Chroma) -> int:
    """Delete every chunk belonging to one document (matched by source). Returns count deleted."""

    matches = vector_store.get(where={"source": source})
    ids = matches["ids"]

    if ids:
        vector_store.delete(ids=ids)

    return len(ids)
