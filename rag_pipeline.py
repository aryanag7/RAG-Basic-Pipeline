from embedder import create_embedding_model
from generator import create_llm, generate_answer
from loader import clean_documents, load_pdf
from retriever import retrieve_relevant_chunks
from splitter import split_documents
from vector_store import create_or_load_vector_store


def run(query: str | None = None) -> None:
    """Run the full RAG pipeline end-to-end."""

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
    if query is None:
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
