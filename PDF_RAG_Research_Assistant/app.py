import json
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    check_ollama,
    ingest_uploaded_pdfs,
    answer_question,
    load_store,
    clear_store,
    list_indexed_documents,
)

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "data" / "uploads"

st.set_page_config(page_title="Local RAG Research Assistant", page_icon="📚", layout="wide")

st.title("📚 AI Research Assistant")
st.caption("Ask questions about your PDFs using local models — no paid API keys.")

with st.sidebar:
    st.header("⚙️ Local model settings")
    llm_model = st.text_input("Chat model", value="llama3.2:3b")
    embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
    st.markdown("---")
    st.subheader("Ollama status")
    ok, message = check_ollama()
    if ok:
        st.success(message)
    else:
        st.error(message)
        st.info("Install Ollama, start it, then pull the two models. See README.md.")
    st.markdown("---")
    st.subheader("Indexed documents")
    docs = list_indexed_documents()
    if docs:
        for name in docs:
            st.write("📄", name)
    else:
        st.caption("No PDFs indexed yet.")
    if st.button("Clear vector index", use_container_width=True):
        clear_store()
        st.session_state.pop("chat_history", None)
        st.success("Vector index cleared.")
        st.rerun()

tab_ingest, tab_chat, tab_about = st.tabs(["1. Upload & index PDFs", "2. Ask questions", "About this project"])

with tab_ingest:
    st.subheader("Add PDF documents")
    st.write("Upload one or more PDFs. Text is extracted page by page, divided into overlapping chunks, embedded locally, and indexed in FAISS.")
    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size (characters)", min_value=300, max_value=1500, value=800, step=100)
    with col2:
        overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=300, value=120, step=20)
    if st.button("Process and index PDFs", type="primary", disabled=not uploads or not ok):
        with st.spinner("Extracting, chunking, embedding and indexing your PDFs..."):
            try:
                result = ingest_uploaded_pdfs(
                    uploads=uploads,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    embedding_model=embedding_model,
                    upload_dir=UPLOAD_DIR,
                )
                st.success(
                    f"Indexed {result['documents']} PDF(s), {result['pages']} page(s), "
                    f"and {result['chunks']} text chunk(s)."
                )
                st.info("You can now open the Ask questions tab.")
            except Exception as exc:
                st.exception(exc)

with tab_chat:
    st.subheader("Ask your documents")
    st.write("Answers are generated from the retrieved PDF passages. Check the source snippets and page numbers.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Sources used"):
                    for src in item["sources"]:
                        st.markdown(
                            f"**{src['source']} — page {src['page']}**  \n"
                            f"> {src['text'][:900]}"
                        )

    question = st.chat_input("Ask a question about your indexed PDFs...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if not ok:
            st.error("Ollama is not available. Check the setup instructions in README.md.")
        elif load_store() is None:
            st.warning("No documents are indexed yet. Upload and process PDFs first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching your PDFs and generating an answer..."):
                    try:
                        result = answer_question(
                            question=question,
                            embedding_model=embedding_model,
                            llm_model=llm_model,
                            top_k=4,
                        )
                        st.markdown(result["answer"])
                        with st.expander("Retrieved source passages"):
                            for src in result["sources"]:
                                st.markdown(
                                    f"**{src['source']} — page {src['page']}**  \n"
                                    f"> {src['text'][:1200]}"
                                )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        })
                    except Exception as exc:
                        st.exception(exc)

with tab_about:
    st.subheader("How this RAG application works")
    st.markdown("""
    1. **PDF extraction:** PyMuPDF extracts text and page numbers.
    2. **Chunking:** each page is split into smaller overlapping passages.
    3. **Embeddings:** Ollama's local `nomic-embed-text` model converts passages into vectors.
    4. **Vector search:** FAISS finds the passages most similar to your question.
    5. **Generation:** the local `llama3.2:3b` model answers using only the retrieved context.
    6. **Citations:** source PDF names and page numbers are displayed with the answer.

    **Privacy:** PDFs, embeddings, and questions stay on your computer when using the local Ollama service.
    """)
    st.warning("This is a learning project. Verify important information against the original PDF.")
