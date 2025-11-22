#!/usr/bin/env python3
import os
import json
import glob
import time
import textwrap
import requests
import faiss
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# -----------------------------------------
# Config
# -----------------------------------------
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL",  "llama3")
DATA_DIR    = os.getenv("RAG_DATA_DIR", "./data")

CHUNK_SIZE   = 800   # characters
CHUNK_OVERLAP = 200  # characters
TOP_K        = 5     # retrieved chunks

# -----------------------------------------
# Helper: Ollama embedding
# -----------------------------------------
def ollama_embed(texts):
    """
    texts: list[str]
    returns: np.ndarray (n, d)
    """
    url = f"{OLLAMA_HOST}/api/embeddings"
    vectors = []
    for t in texts:
        payload = {
            "model": EMBED_MODEL,
            "prompt": t,
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        v = data.get("embedding")
        if v is None:
            raise RuntimeError(f"No embedding in response: {data}")
        vectors.append(np.array(v, dtype="float32"))
    return np.vstack(vectors)


# -----------------------------------------
# Helper: Ollama chat
# -----------------------------------------
def ollama_chat(system_prompt, user_prompt):
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


# -----------------------------------------
# Data loading
# -----------------------------------------
def load_pdf(path):
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            texts.append(txt.strip())
    return "\n".join(texts)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # cheap flatten: stringify
    return json.dumps(obj, indent=2, ensure_ascii=False)


def load_corpus(data_dir):
    """
    Load all PDFs and JSON files from data_dir -> list of (source_id, text).
    """
    docs = []

    pdf_paths = glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
    for p in pdf_paths:
        try:
            txt = load_pdf(p)
            if txt.strip():
                docs.append((p, txt))
        except Exception as e:
            print(f"[WARN] Failed to read PDF {p}: {e}")

    json_paths = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    for p in json_paths:
        try:
            txt = load_json(p)
            if txt.strip():
                docs.append((p, txt))
        except Exception as e:
            print(f"[WARN] Failed to read JSON {p}: {e}")

    return docs


# -----------------------------------------
# Chunking
# -----------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks(docs):
    """
    docs: list[(source_id, full_text)]
    returns:
      chunks: list[dict] with keys: id, source, text
    """
    chunks = []
    cid = 0
    for source, txt in docs:
        for chunk in chunk_text(txt):
            chunk = chunk.strip()
            if not chunk:
                continue
            cid += 1
            chunks.append({
                "id": cid,
                "source": source,
                "text": chunk,
            })
    return chunks


# -----------------------------------------
# Index builder using FAISS
# -----------------------------------------
class RagIndex:
    def __init__(self, dim, index, chunks):
        self.dim = dim
        self.index = index
        self.chunks = chunks

    def search(self, query, k=TOP_K):
        q_vec = ollama_embed([query])[0]  # (d,)
        q_vec = np.expand_dims(q_vec, axis=0)
        D, I = self.index.search(q_vec, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((dist, self.chunks[idx]))
        return results


def build_index(chunks):
    """
    chunks: list[dict] with key 'text'.
    returns RagIndex
    """
    texts = [c["text"] for c in chunks]
    print(f"[INFO] Embedding {len(texts)} chunks...")
    embed_mat = ollama_embed(texts)   # (n, d)
    n, d = embed_mat.shape
    print(f"[INFO] Embedding dim={d}, chunks={n}")

    index = faiss.IndexFlatIP(d)  # inner product
    # normalize for cosine similarity
    faiss.normalize_L2(embed_mat)
    index.add(embed_mat)

    return RagIndex(dim=d, index=index, chunks=chunks)


# -----------------------------------------
# Prompt assembly
# -----------------------------------------
def build_context_prompt(query, retrieved):
    """
    retrieved: list[(dist, chunk_dict)]
    """
    context_blocks = []
    for dist, c in retrieved:
        header = f"[Source: {os.path.basename(c['source'])}, score={dist:.4f}]"
        context_blocks.append(header + "\n" + c["text"])
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the user's question.
If something is not supported by the context, say you don't know.

Context:
{context}

Question:
{query}

Answer clearly:
"""
    return prompt


# -----------------------------------------
# Main REPL
# -----------------------------------------
def main():
    print(f"[INFO] Using Ollama host: {OLLAMA_HOST}")
    print(f"[INFO] EMBED_MODEL={EMBED_MODEL}, CHAT_MODEL={CHAT_MODEL}")
    print(f"[INFO] Loading data from: {DATA_DIR}")

    docs = load_corpus(DATA_DIR)
    if not docs:
        print("[ERROR] No PDF/JSON files found. Put them into ./data or set RAG_DATA_DIR.")
        return

    print(f"[INFO] Loaded {len(docs)} documents.")
    chunks = build_chunks(docs)
    print(f"[INFO] Built {len(chunks)} text chunks.")

    rag_index = build_index(chunks)
    print("[INFO] RAG index ready.")

    system_prompt = (
        "You answer user questions based on the provided context chunks. "
        "If the context is insufficient, say you don't know."
    )

    while True:
        try:
            q = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("[INFO] Bye.")
            break

        t0 = time.time()
        retrieved = rag_index.search(q, k=TOP_K)
        if not retrieved:
            print("No context available.")
            continue

        prompt = build_context_prompt(q, retrieved)
        answer = ollama_chat(system_prompt, prompt)
        dt = time.time() - t0

        print("\nAssistant> " + answer)
        print(f"\n[DEBUG] Retrieved {len(retrieved)} chunks in {dt:.2f}s.")
        for dist, c in retrieved:
            print(f"  - {os.path.basename(c['source'])}: score={dist:.4f}, chunk_id={c['id']}")


if __name__ == "__main__":
    main()

