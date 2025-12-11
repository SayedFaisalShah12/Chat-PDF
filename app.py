# app.py
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import faiss
import requests
import json
from typing import List

# ---------------------------
# CONFIG
# ---------------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 4
BATCH_SIZE = 16

st.set_page_config(page_title="Chat PDF (RAG) - Ollama", page_icon="📄")
st.title("📄 Chat with PDFs (RAG + Ollama + Llama 3)")


# ----------------------------------------------------
# PDF → TEXT
# ----------------------------------------------------
def extract_text(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            st.warning(f"Failed to read {pdf.name}: {e}")
    return text


# ----------------------------------------------------
# CHUNKING
# ----------------------------------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# ----------------------------------------------------
# OLLAMA EMBEDDINGS
# ----------------------------------------------------
def embed_text(batch: List[str]):
    payload = {
        "model": EMBED_MODEL,
        "input": batch
    }
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json=payload
    )
    response.raise_for_status()
    data = response.json()

    # Format: {"data":[{"embedding": [...]}, ...]}
    vectors = [item["embedding"] for item in data["data"]]
    return np.array(vectors, dtype="float32")


# ----------------------------------------------------
# NORMALIZE (cosine similarity)
# ----------------------------------------------------
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


# ----------------------------------------------------
# BUILD FAISS INDEX
# ----------------------------------------------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ----------------------------------------------------
# RETRIEVAL
# ----------------------------------------------------
def retrieve(query, chunks, index):
    q_emb = embed_text([query])
    q_emb = normalize(q_emb)

    D, I = index.search(q_emb, TOP_K)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    return retrieved


# ----------------------------------------------------
# GENERATION (Llama 3)
# ----------------------------------------------------
def generate_answer(prompt):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload
    )
    response.raise_for_status()
    output = response.json()
    return output.get("response", "")


# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embs" not in st.session_state:
    st.session_state.embs = None

if "index" not in st.session_state:
    st.session_state.index = None

if "ready" not in st.session_state:
    st.session_state.ready = False


# ----------------------------------------------------
# SIDEBAR UI
# ----------------------------------------------------
with st.sidebar:
    st.header("📥 Upload PDFs")
    pdfs = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Process PDFs"):
        if not pdfs:
            st.error("Upload PDF files first!")
        else:
            st.info("Extracting text...")
            text = extract_text(pdfs)

            st.info("Chunking text...")
            chunks = chunk_text(text)
            st.write(f"Created {len(chunks)} text chunks")

            # Embed in batches
            all_vecs = []
            st.info("Embedding chunks with Ollama…")

            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                try:
                    vec = embed_text(batch)
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
                    break
                all_vecs.append(vec)

            if all_vecs:
                embs = np.vstack(all_vecs)
                embs = normalize(embs)
                index = build_index(embs)

                st.session_state.chunks = chunks
                st.session_state.embs = embs
                st.session_state.index = index
                st.session_state.ready = True

                st.success("PDFs processed successfully! Ask questions now.")


# ----------------------------------------------------
# MAIN CHAT SECTION
# ----------------------------------------------------
st.header("💬 Ask Your Question")

question = st.text_input("Your question:")

if st.button("Ask"):
    if not st.session_state.ready:
        st.error("You must upload & process PDFs first!")
    else:
        st.info("Retrieving relevant chunks…")
        retrieved = retrieve(question, st.session_state.chunks, st.session_state.index)

        context = "\n\n---\n\n".join(retrieved)

        prompt = f"""
Use the following document context to answer the question.
If the answer is not found, say "I don't know".

Context:
{context}

Question: {question}

Answer:
        """

        st.info("Generating answer using Llama 3…")
        answer = generate_answer(prompt)

        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📄 Retrieved Chunks")
        for c in retrieved:
            st.write("- " + c[:500] + "...")
