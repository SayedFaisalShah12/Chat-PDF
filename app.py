# app.py
import streamlit as st
from PyPDF2 import PdfReader
import requests
import numpy as np
import faiss
import json
from typing import List

hi
# ------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"     # embedding model
LLM_MODEL = "llama3"                  # generation model
EMBED_BATCH = 16
TOP_K = 4
# -------------------------------------

st.set_page_config(page_title="Chat PDF - RAG with Ollama", page_icon="📄")
st.title("📄 Chat PDF — RAG with Ollama (No LangChain)")

# -------------------------------------
# Extract Text from PDF
# -------------------------------------
def extract_text_from_pdfs(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        except:
            st.warning(f"Failed to read {pdf.name}")
    return text


# -------------------------------------
# Text Chunking
# -------------------------------------
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# -------------------------------------
# Ollama Embedding
# -------------------------------------
def embed_texts(texts: List[str], model=EMBED_MODEL):
    url = f"{OLLAMA_URL}/api/embed"
    payload = {"model": model, "input": texts}

    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()

    # ensure multi-vector output
    if "data" in data:
        return np.array([d["embedding"] for d in data["data"]], dtype=np.float32)

    if "embedding" in data:  # single case
        return np.array([data["embedding"]], dtype=np.float32)

    raise RuntimeError("Unexpected embedding response:", data)


# -------------------------------------
# Normalize
# -------------------------------------
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# -------------------------------------
# Build FAISS Index
# -------------------------------------
def build_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


# -------------------------------------
# Retrieve Top-K
# -------------------------------------
def retrieve(query, chunks, index, top_k=TOP_K):
    q_emb = embed_texts([query])
    q_emb = normalize(q_emb)

    D, I = index.search(q_emb, top_k)
    results = []

    for idx in I[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results


# -------------------------------------
# Ollama LLM Generate
# -------------------------------------
def generate(prompt, model=LLM_MODEL):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt}

    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()

    if "response" in data:
        return data["response"]

    return str(data)


# ------------- SESSION STATE ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss" not in st.session_state:
    st.session_state.faiss = None
if "ready" not in st.session_state:
    st.session_state.ready = False

# ------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📤 Upload PDFs")
    pdfs = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

    embed_model = st.text_input("Embedding Model", value=EMBED_MODEL)
    llm_model = st.text_input("LLM Model", value=LLM_MODEL)
    top_k = st.number_input("Top-K Chunks", min_value=1, max_value=10, value=TOP_K)

    if st.button("⚙️ Process PDFs"):
        if not pdfs:
            st.error("Please upload PDF files first.")
        else:
            st.info("Extracting text...")
            text = extract_text_from_pdfs(pdfs)

            chunks = chunk_text(text)
            st.success(f"Created {len(chunks)} chunks")

            # embed chunks
            st.info("Computing embeddings (this may take some time)...")
            all_embs = []

            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i:i+EMBED_BATCH]
                try:
                    batch_emb = embed_texts(batch, model=embed_model)
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
                    batch_emb = None
                all_embs.append(batch_emb)

            embs = np.vstack(all_embs)
            embs = normalize(embs)

            index = build_index(embs)

            st.session_state.chunks = chunks
            st.session_state.embeddings = embs
            st.session_state.faiss = index
            st.session_state.ready = True

            st.success("PDFs processed successfully! You can now ask questions.")


# ------------- MAIN CHAT UI ----------------
st.subheader("💬 Ask a question about your PDFs")

query = st.text_input("Your question")

if st.button("Ask"):
    if not st.session_state.ready:
        st.error("Please upload and process PDFs first.")
    else:
        st.info("Retrieving relevant chunks...")
        results = retrieve(query, st.session_state.chunks, st.session_state.faiss, top_k=int(top_k))

        st.markdown("### 🔎 Retrieved Context")
        for i, r in enumerate(results, 1):
            st.write(f"**Chunk {i}:** {r[:300]}...")

        prompt = f"""
Use ONLY the following context to answer the question.

Context:
{chr(10).join(results)}

Question: {query}

If answer is not found, say "I don't know".
"""

        st.info("Generating answer with Llama-3...")
        answer = generate(prompt, model=llm_model)

        st.markdown("### 🧠 Answer")
        st.write(answer)
