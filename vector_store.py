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

def create_or_load_vector_store(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """Create Chroma and add only chunks not already stored."""

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_STORE_PATH),
        collection_configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

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

        print(f"Added {len(new_chunks)} chunks to Chroma.")
    else:
        print("All chunks are already stored in Chroma.")

    print(
        f"Total records in vector store: "
        f"{len(vector_store.get()['ids'])}"
    )

    return vector_store
