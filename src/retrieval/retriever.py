import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
VECTOR_STORE_FOLDER = "data/vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_FOLDER, "faiss_index.bin")
EMBEDDINGS_METADATA_PATH = os.path.join(VECTOR_STORE_FOLDER, "embeddings_metadata.json")

# -----------------------------
# Load FAISS index
# -----------------------------
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)
print(f"FAISS index loaded. Total vectors: {index.ntotal}")

# -----------------------------
# Load embeddings metadata
# -----------------------------
with open(EMBEDDINGS_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Loaded metadata for {len(metadata)} chunks.")

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Retriever function
# -----------------------------
def retrieve(query, top_k=5):
    """
    Retrieve the top_k most relevant chunks for a user query.
    
    Args:
        query (str): The user's question
        top_k (int): Number of chunks to return

    Returns:
        List of dicts: Each dict contains 'text', 'source_file', 'chunk_index', 'num_words'
    """
    # Convert query to embedding
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    
    # Search FAISS index
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx in indices[0]:
        chunk_info = metadata[idx].copy()
        
        # Load the actual chunk text
        chunk_file = chunk_info["chunk_file"]
        chunk_path = os.path.join("data/processed/chunks", chunk_file)
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()
        
        chunk_info["text"] = chunk_text
        results.append(chunk_info)
    
    return results

# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    query = "What are the main cybersecurity threats?"
    top_chunks = retrieve(query, top_k=3)
    
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk['source_file']} | Chunk: {chunk['chunk_index']}")
        print(chunk["text"][:500] + "...")