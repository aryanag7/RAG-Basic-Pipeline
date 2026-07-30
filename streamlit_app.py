import logging
from hashlib import sha256

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from embedder import create_embedding_model
from generator import create_llm, generate_answer
from loader import clean_documents, load_pdf, load_pdf_from_bytes
from retriever import retrieve_relevant_chunks
from splitter import split_documents
from vector_store import add_new_chunks, create_or_load_vector_store

# Streamlit's file watcher walks every imported module's __path__ to find local
# files to auto-reload on save. transformers (pulled in via langchain-huggingface)
# lazily imports optional vision submodules that need torchvision, which isn't
# installed here; the resulting ModuleNotFoundError is caught internally and only
# logged as noise. Silence it rather than adding an unused torchvision dependency.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


@st.cache_resource(show_spinner="Loading document and building index...")
def load_vector_store() -> Chroma:
    """Build/load the vector store once per app process."""

    documents = load_pdf()
    cleaned = clean_documents(documents)
    chunks = split_documents(cleaned)
    embedding_model = create_embedding_model()

    return create_or_load_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
    )


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatOpenAI:
    """Create the OpenAI model once per app process."""

    return create_llm()


st.set_page_config(page_title="System Design RAG")
st.title("System Design RAG")
st.caption("Ask questions about the system-design PDF and get grounded answers.")

with st.sidebar:
    st.markdown("### Document source")
    st.caption("Currently indexed: `data/System Design Concepts.pdf` (fixed)")

    uploaded_files = st.file_uploader(
        "Upload additional PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded PDFs are added to the same knowledge base used to answer questions.",
    )

    processed = st.session_state.setdefault("processed_uploads", {})

    if uploaded_files:
        vector_store = load_vector_store()

        for uploaded_file in uploaded_files:
            data = uploaded_file.getvalue()
            content_hash = sha256(data).hexdigest()

            if content_hash in processed:
                continue

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    docs = load_pdf_from_bytes(data, filename=f"upload:{content_hash}")
                    cleaned = clean_documents(docs)
                    chunks = split_documents(cleaned)
                    added = add_new_chunks(chunks, vector_store=vector_store) if chunks else 0
                except Exception as exc:
                    processed[content_hash] = {
                        "name": uploaded_file.name,
                        "status": "error",
                        "detail": str(exc),
                    }
                else:
                    status = "empty" if not chunks else ("added" if added else "duplicate")
                    processed[content_hash] = {
                        "name": uploaded_file.name,
                        "status": status,
                        "added": added,
                    }

    for info in processed.values():
        if info["status"] == "added":
            st.success(f"{info['name']}: {info['added']} chunks added.")
        elif info["status"] == "duplicate":
            st.info(f"{info['name']}: already indexed.")
        elif info["status"] == "empty":
            st.warning(f"{info['name']}: no extractable text.")
        else:
            st.error(f"{info['name']}: {info['detail']}")

query = st.text_input(
    "Ask a system-design question",
    placeholder="e.g. What is a load balancer?",
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            vector_store = load_vector_store()
            results = retrieve_relevant_chunks(query=query, vector_store=vector_store)

            if results:
                llm = load_llm()
                answer = generate_answer(query=query, results=results, llm=llm)
            else:
                answer = None

        st.markdown("### Answer")
        if answer is not None:
            st.write(answer)
        else:
            st.info("I could not find relevant information in the provided document.")
