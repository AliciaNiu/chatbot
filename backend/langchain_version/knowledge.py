import faiss
import numpy as np
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# --- Embedding helper ---
def get_embedding(text: str, model="text-embedding-3-small"):
    resp = openai.Embedding.create(input=[text], model=model)
    return np.array(resp["data"][0]["embedding"], dtype="float32")

# --- Build FAISS index from text chunks ---
def build_index(text_chunks):
    dim = len(get_embedding("test"))  # dimension of embeddings
    index = faiss.IndexFlatL2(dim)

    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    index.add(np.array(embeddings))

    return index, text_chunks

# --- Query KB ---
def query_index(index, text_chunks, query, top_k=3):
    q_emb = get_embedding(query)
    D, I = index.search(np.array([q_emb]), top_k)
    return [text_chunks[i] for i in I[0]]
