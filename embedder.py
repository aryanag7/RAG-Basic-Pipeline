from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def inspect_one_embedding(chunks, embedding_model) -> None:
    """Create and inspect the vector for one chunk."""

    first_chunk_text = chunks[0].page_content

    vectors = embedding_model.embed_documents([first_chunk_text])
    vector = vectors[0]

    print("\n" + "=" * 60)
    print("Embedding inspection")
    print(f"Chunk characters: {len(first_chunk_text)}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 10 numbers: {vector[:10]}")
