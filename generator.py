from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL_NAME


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
