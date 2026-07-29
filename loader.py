import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import PDF_PATH


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

    for raw_line in text.replace(" ", " ").splitlines():
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


def inspect_documents(documents):
    """Print sample content to check whether parsing worked correctly."""

    print(f"Number of Document objects: {len(documents)}")

    for document in documents[:3]:
        print("\n" + "=" * 60)
        print("Metadata:")
        print(document.metadata)

        print("\nExtracted text:")
        print(document.page_content[:700])
