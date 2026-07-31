import logging
import time
from hashlib import sha256

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

import ui_theme
from embedder import create_embedding_model
from generator import create_llm, generate_answer
from loader import clean_documents, load_pdf_from_bytes
from retriever import retrieve_relevant_chunks
from splitter import split_documents
from vector_store import (
    add_new_chunks,
    create_or_load_vector_store,
    delete_source,
    list_indexed_documents,
)

# Streamlit's file watcher walks every imported module's __path__ to find local
# files to auto-reload on save. transformers (pulled in via langchain-huggingface)
# lazily imports optional vision submodules that need torchvision, which isn't
# installed here; the resulting ModuleNotFoundError is caught internally and only
# logged as noise. Silence it rather than adding an unused torchvision dependency.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


@st.cache_resource(show_spinner="Connecting to the document index...")
def load_vector_store() -> Chroma:
    """Open the persistent vector store once per app process."""

    embedding_model = create_embedding_model()

    return create_or_load_vector_store(embedding_model=embedding_model)


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatOpenAI:
    """Create the OpenAI model once per app process."""

    return create_llm()


st.set_page_config(page_title="SourceLens")
st.html(ui_theme.CUSTOM_CSS)
with st.container(key="app_header"):
    st.title("SourceLens")
    st.caption(
        "Upload PDFs, search across your document library, and get clear answers "
        "grounded in the original sources."
    )

vector_store = load_vector_store()
processed = st.session_state.setdefault("processed_uploads", {})

documents = list_indexed_documents(vector_store)
doc_count = len(documents)
total_chunks = sum(document["chunk_count"] for document in documents)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents indexed", doc_count, border=True)
with col2:
    st.metric("Chunks in index", f"{total_chunks:,}", border=True)
with col3:
    with st.container(border=True):
        st.badge("Hybrid search", icon=":material/hub:", color="violet")
        st.caption("BM25 keyword + embedding similarity, fused with RRF")

st.markdown("#### Indexed documents")
if not documents:
    st.caption("No documents indexed yet. Upload a PDF to get started.")
else:
    header_cols = st.columns([4, 2, 2, 1])
    for col, label in zip(header_cols, ["Document", "Size", "Chunks", ""]):
        col.markdown(f"**{label}**")

    for document in documents:
        row_cols = st.columns([4, 2, 2, 1])
        row_cols[0].write(document["filename"])
        row_cols[1].write(ui_theme.format_file_size(document["size"]))
        row_cols[2].write(str(document["chunk_count"]))
        if row_cols[3].button(
            "🗑️",
            key=f"delete_{document['source']}",
            help=f"Delete {document['filename']}",
        ):
            delete_source(document["source"], vector_store)
            processed.pop(document["source"].removeprefix("upload:"), None)
            st.rerun()

with st.sidebar:
    st.markdown("### Document source")
    if doc_count == 0:
        st.caption("Upload a PDF to get started.")
    else:
        st.caption(f"{doc_count} document(s) indexed.")

    uploaded_files = st.file_uploader(
        "Upload additional PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded PDFs are added to the same knowledge base used to answer questions.",
    )

    if uploaded_files:
        newly_processed = False

        for uploaded_file in uploaded_files:
            data = uploaded_file.getvalue()
            content_hash = sha256(data).hexdigest()

            if content_hash in processed:
                continue

            newly_processed = True

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    docs = load_pdf_from_bytes(data, filename=f"upload:{content_hash}")
                    for doc in docs:
                        doc.metadata["filename"] = uploaded_file.name
                        doc.metadata["file_size"] = len(data)
                    cleaned = clean_documents(docs)
                    chunks = split_documents(cleaned)
                    added = add_new_chunks(chunks, vector_store=vector_store) if chunks else 0
                except Exception as exc:
                    processed[content_hash] = {
                        "name": uploaded_file.name,
                        "status": "error",
                        "detail": str(exc),
                        "size": len(data),
                    }
                else:
                    status = "empty" if not chunks else ("added" if added else "duplicate")
                    processed[content_hash] = {
                        "name": uploaded_file.name,
                        "status": status,
                        "added": added,
                        "size": len(data),
                    }

        if newly_processed:
            st.rerun()

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
    "Ask a question about your documents",
    placeholder="e.g. What does the document say about load balancing?",
)
ask_clicked = st.button("Ask", type="primary", disabled=(doc_count == 0))
if doc_count == 0:
    st.caption("Upload a PDF to get started.")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.", icon=":material/edit_note:")
    else:
        pipeline_slot = st.empty()

        pipeline_slot.html(ui_theme.render_pipeline_frame("retrieve"))
        results = retrieve_relevant_chunks(query=query, vector_store=vector_store)

        if results:
            pipeline_slot.html(ui_theme.render_pipeline_frame("generate"))
            llm = load_llm()
            answer = generate_answer(query=query, results=results, llm=llm)
            pipeline_slot.html(ui_theme.render_pipeline_frame("complete"))
            time.sleep(0.4)
        else:
            answer = None

        pipeline_slot.empty()

        st.markdown("### Answer")
        if answer is not None:
            with st.container(border=True, key="answer_card"):
                st.write(answer)
        else:
            st.info(
                "I could not find relevant information in the provided document.",
                icon=":material/search_off:",
            )
