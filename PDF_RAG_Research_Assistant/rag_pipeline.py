import json
import re
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import ollama
import fitz  # PyMuPDF

APP_DIR = Path(__file__).resolve().parent
STORE_DIR = APP_DIR / "vector_store"
INDEX_PATH = STORE_DIR / "index.faiss"
METADATA_PATH = STORE_DIR / "metadata.json"


def check_ollama():
    """Return whether the local Ollama server can be reached."""
    try:
        ollama.list()
        return True, "Ollama is running."
    except Exception as exc:
        return False, f"Cannot connect to Ollama: {exc}"


def extract_pdf_pages(pdf_bytes: bytes, filename: str):
    """Extract non-empty text from a PDF, preserving 1-based page numbers."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": text,
                })
    return pages


def split_text(text: str, chunk_size: int = 800, overlap: int = 120):
    """Split text into overlapping character windows, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size.")

    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + int(chunk_size * 0.6):
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _embed_texts(texts: list[str], model: str):
    """Create local embeddings through the Ollama embedding model."""
    response = ollama.embed(model=model, input=texts)
    embeddings = np.asarray(response["embeddings"], dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError("The embedding model returned an unexpected embedding shape.")
    faiss.normalize_L2(embeddings)
    return embeddings


def _read_metadata():
    if not METADATA_PATH.exists():
        return []
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_store(index, metadata):
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_store():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None
    index = faiss.read_index(str(INDEX_PATH))
    metadata = _read_metadata()
    if index.ntotal != len(metadata):
        raise RuntimeError("FAISS index and metadata are out of sync. Clear and rebuild the index.")
    return index, metadata


def clear_store():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (INDEX_PATH, METADATA_PATH):
        if path.exists():
            path.unlink()


def list_indexed_documents():
    metadata = _read_metadata()
    return sorted({item["source"] for item in metadata})


def ingest_uploaded_pdfs(uploads, chunk_size, overlap, embedding_model, upload_dir: Path):
    upload_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    total_pages = 0
    document_count = 0

    for uploaded_file in uploads:
        filename = Path(uploaded_file.name).name
        pdf_bytes = uploaded_file.getvalue()
        if not pdf_bytes:
            continue

        # Keep an original copy in the project's data/uploads folder.
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        (upload_dir / safe_name).write_bytes(pdf_bytes)

        pages = extract_pdf_pages(pdf_bytes, filename)
        if not pages:
            raise ValueError(
                f"No extractable text was found in {filename}. "
                "It may be a scanned PDF; OCR is not included in this first version."
            )

        document_count += 1
        total_pages += len(pages)
        for page in pages:
            for chunk in split_text(page["text"], chunk_size, overlap):
                all_chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk,
                })

    if not all_chunks:
        raise ValueError("No text chunks were created from the uploaded PDFs.")

    texts = [item["text"] for item in all_chunks]
    embeddings = _embed_texts(texts, embedding_model)

    existing = load_store()
    if existing:
        old_index, old_metadata = existing
        if old_index.d != embeddings.shape[1]:
            raise ValueError(
                "The new embedding model has a different vector dimension from the existing index. "
                "Clear the index before changing embedding models."
            )
        old_index.add(embeddings)
        all_metadata = old_metadata + all_chunks
        index = old_index
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        all_metadata = all_chunks

    _save_store(index, all_metadata)
    return {
        "documents": document_count,
        "pages": total_pages,
        "chunks": len(all_chunks),
    }


def answer_question(question: str, embedding_model: str, llm_model: str, top_k: int = 4):
    stored = load_store()
    if stored is None:
        raise RuntimeError("No vector index exists. Upload and index PDFs first.")
    index, metadata = stored

    query_vector = _embed_texts([question], embedding_model)
    k = min(max(1, top_k), index.ntotal)
    scores, positions = index.search(query_vector, k)

    sources = []
    context_parts = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        item = metadata[int(position)]
        source = {
            "source": item["source"],
            "page": item["page"],
            "text": item["text"],
            "score": float(score),
        }
        sources.append(source)
        context_parts.append(
            f"[Source: {item['source']}, page {item['page']}]\n{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are a careful research assistant. Answer the user's question using only "
        "the supplied document context. If the context does not contain the answer, "
        "say clearly that the uploaded documents do not provide enough information. "
        "Do not invent facts or citations. Cite supporting passages inline using "
        "[filename, page N]. Keep the answer clear and useful."
    )
    user_prompt = (
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer from the document context and include page citations."
    )

    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return {
        "answer": response["message"]["content"],
        "sources": sources,
    }
