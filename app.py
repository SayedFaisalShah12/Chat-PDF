# app.py
import streamlit as st
from PyPDF2 import PdfReader
import requests
import numpy as np
import faiss
import math
import json
from typing import List

# ---------- CONFIG ----------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"   # change to the embedding model you pulled
LLM_MODEL = "llama2"                # change to your generation model
EMBED_BATCH_SIZE = 16
TOP_K = 4
# ----------------------------

st.set_page_config(page_title="RAG with Ollama (no LangChain)", page_icon="📚")
st.title("Chat-PDf (RAG)")

# ---------- Helpers ----------
def extract_text_from_pdfs(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                p = page.extract_text()
                if p:
                    text += p + "\n"
        except Exception as e:
            st.warning(f"Could not read {getattr(pdf, 'name', 'uploaded file')}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Simple chunker by characters with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def call_ollama_embed(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Call Ollama embedding endpoint for a list of strings.
       Returns a 2D numpy array (n_texts, dim).
    """
    # Ollama embed endpoint expects JSON like {"model":"...","input":["str1","str2",...]}
    url = f"{OLLAMA_URL}/api/embed"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Ollama's embed response format varies by version. Try common keys:
    # Expect something like: {"embedding": [...]} or {"data":[{"embedding":[...]}], ...}
    # We handle a few formats robustly:
    embeddings = []
    if isinstance(data, dict):
        # case: {"embedding": [..]} for single input
        if "embedding" in data and isinstance(data["embedding"][0], list) is False:
            # single vector returned for single input; handle single-case
            return np.array([data["embedding"]], dtype=np.float32)
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if "embedding" in item:
                    embeddings.append(item["embedding"])
        elif "embeddings" in data:
            embeddings = data["embeddings"]
        elif "embedding" in data and isinstance(data["embedding"], list):
            # maybe list of vectors
            embeddings = data["embedding"]
    elif isinstance(data, list):
        # some versions return a list of vectors directly
        embeddings = data

    if not embeddings:
        # last attempt: try to parse keys
        if "response" in data:
            # maybe nested
            try:
                embeddings = [r["embedding"] for r in data["response"] if "embedding" in r]
            except Exception:
                pass

    if not embeddings:
        raise RuntimeError(f"Unexpected embed response structure from Ollama: {data}")

    arr = np.array(embeddings, dtype=np.float32)
    return arr

def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length (for cosine similarity with IndexFlatIP)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product; use normalized embeddings for cosine
    index.add(embeddings)
    return index

def retrieve_top_k(query: str, chunks: List[str], index: faiss.IndexFlatIP,
                   chunk_embeddings: np.ndarray, top_k: int = TOP_K) -> List[str]:
    q_emb = call_ollama_embed([query])  # shape (1, dim)
    q_emb = normalize_embeddings(q_emb)
    D, I = index.search(q_emb, top_k)
    ids = I[0].tolist()
    results = []
    for idx in ids:
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(chunks[idx])
    return results

def call_ollama_generate(prompt: str, model: str = LLM_MODEL, temperature: float = 0.2, max_tokens: int = 256):
    """Call Ollama generate endpoint. Returns text output."""
    url = f"{OLLAMA_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Ollama may return in different structures; try common ones:
    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"]
        if "responses" in data and isinstance(data["responses"], list):
            # join responses
            texts = []
            for r in data["responses"]:
                if isinstance(r, dict) and "content" in r:
                    texts.append(r["content"])
                elif isinstance(r, str):
                    texts.append(r)
            return " ".join(texts).strip()
        # some versions: {"outputs":[{"content":"..."}]}
        if "outputs" in data and isinstance(data["outputs"], list):
            out = []
            for o in data["outputs"]:
                if isinstance(o, dict) and "content" in o:
                    out.append(o["content"])
            return " ".join(out).strip()
    # fallback: return full JSON as string
    return json.dumps(data)

# ---------- Streamlit state ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Sidebar: upload & process PDFs ----------
with st.sidebar:
    st.subheader("Upload PDFs for RAG")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    embed_model_input = st.text_input("Embedding model (ollama)", value=EMBED_MODEL)
    llm_model_input = st.text_input("LLM model (ollama)", value=LLM_MODEL)
    top_k_input = st.number_input("Top-k retrieval", min_value=1, max_value=10, value=TOP_K)
    if st.button("Process PDFs"):
        if not uploaded_files:
            st.error("Please upload 1 or more PDFs.")
        else:
            st.info("Extracting text from PDFs...")
            raw_text = extract_text_from_pdfs(uploaded_files)
            if not raw_text.strip():
                st.error("No text extracted from PDFs.")
            else:
                st.info("Chunking text...")
                chunks = chunk_text(raw_text, chunk_size=800, chunk_overlap=200)
                st.write(f"Created {len(chunks)} chunks.")
                # embed in batches
                st.info("Computing embeddings via Ollama (this may take a while)...")
                all_embs = []
                for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                    batch = chunks[i:i+EMBED_BATCH_SIZE]
                    try:
                        batch_emb = call_ollama_embed(batch, model=embed_model_input)
                    except Exception as e:
                        st.error(f"Embedding call failed: {e}")
                        all_embs = []
                        break
                    all_embs.append(batch_emb)
                if not all_embs:
                    st.error("Embedding failed — check your Ollama server and embed model.")
                else:
                    embs = np.vstack(all_embs)
                    embs = normalize_embeddings(embs)
                    index = build_faiss_index(embs)
                    # save to state
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embs
                    st.session_state.faiss_index = index
                    st.session_state.processed = True
                    st.session_state.chat_history = []
                    st.session_state.top_k = int(top_k_input)
                    st.success("Processing complete. You can ask questions now.")

# ---------- Main UI: ask questions ----------
st.subheader("Ask a question about uploaded PDFs")
question = st.text_input("Your question:")

if st.button("Ask") and question:
    if not st.session_state.processed:
        st.warning("Upload and Process PDFs first in the sidebar.")
    else:
        with st.spinner("Retrieving relevant context..."):
            try:
                top_k = st.session_state.get("top_k", TOP_K)
                relevant = retrieve_top_k(question, st.session_state.chunks,
                                          st.session_state.faiss_index, st.session_state.embeddings,
                                          top_k=top_k)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                relevant = []
        context_text = "\n\n---\n\n".join(relevant)
        prompt = f"Use the following context from documents to answer the question. If the answer is not contained, say you don't know.\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer concisely:"
        st.markdown("**Context (retrieved):**")
        for i, r in enumerate(relevant, start=1):
            st.write(f"**Chunk {i}:** {r[:800]}{'...' if len(r)>800 else ''}")
        with st.spinner("Generating answer from Ollama..."):
            try:
                answer = call_ollama_generate(prompt, model=llm_model_input, temperature=0.2, max_tokens=256)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                answer = None
        if answer:
            st.markdown("### Answer")
            st.write(answer)
            # store chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})
        else:
            st.error("No answer returned.")

# show conversation history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## Conversation History")
    for item in st.session_state.chat_history[::-1]:
        st.write(f"**Q:** {item['question']}")
        st.write(f"**A:** {item['answer']}")
