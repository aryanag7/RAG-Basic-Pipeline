import logging

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from embedder import create_embedding_model
from generator import create_llm, generate_answer
from loader import clean_documents, load_pdf
from retriever import retrieve_relevant_chunks
from splitter import split_documents
from vector_store import create_or_load_vector_store

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
    st.file_uploader(
        "Upload a different PDF",
        type=["pdf"],
        disabled=True,
        help=(
            "Coming soon — uploading is not yet wired up. "
            "The app currently always answers from the bundled PDF above."
        ),
    )

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
