import re


from pathlib import Path
from hashlib import sha256

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


PDF_PATH = Path("data") / "System Design Concepts.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = Path("vector_store")
COLLECTION_NAME = "system_design_concepts"
TOP_K = 5
MAX_DISTANCE = 0.65
OPENAI_MODEL_NAME = "gpt-5-mini"


def load_pdf():
    """Load the PDF into LangChain Document objects."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF was not found at: {PDF_PATH.resolve()}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    return documents

def is_list_item(line: str) -> bool:
    """Check whether a line begins with a bullet or number."""

    return bool(
        re.match(r"^(?:[•●▪◦*-]|\d+[.)])\s*", line)
    )


def looks_like_heading(line: str) -> bool:
    """Conservatively identify short headings."""

    words = line.split()

    return (
        1 <= len(words) <= 8
        and line[0].isupper()
        and not line.endswith((".", ",", ";"))
    )


def clean_text(text: str) -> str:
    """Remove accidental spacing while preserving meaningful lines."""

    normalised_lines = []

    for raw_line in text.replace("\u00a0", " ").splitlines():
        # Convert repeated spaces and tabs into one space.
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if line:
            normalised_lines.append(line)

    cleaned_lines = []

    for line in normalised_lines:
        word_count = len(line.split())

        # PyPDF sometimes extracts a sentence like:
        # some
        # requests
        # are
        # heavy.
        #
        # Join those small lowercase fragments back together.
        is_small_fragment = (
            word_count <= 2
            and not is_list_item(line)
            and not looks_like_heading(line)
        )

        if is_small_fragment and cleaned_lines:
            cleaned_lines[-1] = f"{cleaned_lines[-1]} {line}"
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean every Document while preserving its metadata."""

    return [
        Document(
            page_content=clean_text(document.page_content),
            metadata=document.metadata.copy(),
        )
        for document in documents
    ]

def split_documents(documents):
    """Split page Documents into smaller chunk Documents."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

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

def create_llm() -> ChatOpenAI:
    """Create the OpenAI model used to generate the final answer."""

    return ChatOpenAI(
        model=OPENAI_MODEL_NAME,
    )

def build_context(
    results: list[tuple[Document, float]],
) -> str:
    """Combine retrieved chunks into one context string."""

    context_parts = []

    for index, (document, distance) in enumerate(results, start=1):
        page = document.metadata.get("page_label", "Unknown")

        context_part = (
            f"[Context {index} | Page {page}]\n"
            f"{document.page_content}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)

def generate_answer(
    query: str,
    results: list[tuple[Document, float]],
    llm: ChatOpenAI,
) -> str:
    """Generate an answer using only the retrieved document context."""

    context = build_context(results)

    messages = [
        (
            "system",
            """
            You are a helpful system-design tutor.

            Answer the user's question using only the provided document context.

            Rules:
            1. Do not use outside knowledge.
            2. If the context does not contain enough information, say:
            "I could not find enough information in the provided document."
            3. Explain the answer in simple English.
            4. Include the relevant page number when possible.
            5. Do not mention information that is not supported by the context.
            """,
                    ),
                    (
                        "human",
                        f"""
            Question:
            {query}

            Document context:
            {context}
            """,
        ),
    ]

    response = llm.invoke(messages)

    return response.content


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


def inspect_documents(documents):
    """Print sample content to check whether parsing worked correctly."""

    print(f"Number of Document objects: {len(documents)}")

    for document in documents[:3]:
        print("\n" + "=" * 60)
        print("Metadata:")
        print(document.metadata)

        print("\nExtracted text:")
        print(document.page_content[:700])


if __name__ == "__main__":
    # 1. Load the PDF.
    loaded_documents = load_pdf()

    # 2. Clean the PDF text.
    cleaned_documents = clean_documents(loaded_documents)

    # 3. Split the documents into chunks.
    chunks = split_documents(cleaned_documents)

    print(f"Pages loaded: {len(loaded_documents)}")
    print(f"Chunks created: {len(chunks)}")

    # 4. Load the local embedding model.
    embedding_model = create_embedding_model()

    # 5. Create or load the persistent Chroma vector store.
    vector_store = create_or_load_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
    )

    # 6. Ask the user for a question.
    query = input("\nAsk a system-design question: ").strip()

    if query:
        # 7. Retrieve the top five chunks and apply the threshold.
        results = retrieve_relevant_chunks(
            query=query,
            vector_store=vector_store,
        )

        if results:
            print(f"\nRelevant chunks found: {len(results)}")

            # 8. Create the OpenAI model.
            llm = create_llm()

            # 9. Send the question and retrieved context to the LLM.
            answer = generate_answer(
                query=query,
                results=results,
                llm=llm,
            )

            # 10. Display the final answer.
            print("\n" + "=" * 60)
            print("Answer:")
            print("=" * 60)
            print(answer)

        else:
            # The OpenAI API is not called when no relevant chunks pass
            # the distance threshold.
            print(
                "\nI could not find relevant information "
                "in the provided document."
            )
    else:
        print("No question was entered.")