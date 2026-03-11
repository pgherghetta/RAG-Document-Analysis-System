# src/generation/answer_generator.py

import os
import sys
import json
import numpy as np
import faiss

# -----------------------------
# Prevent multiprocessing crashes on macOS
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add parent folder to sys.path to import retriever
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retriever import retrieve
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# OpenAI API setup
# -----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set your OPENAI_API_KEY in your environment variables.")

client = OpenAI(api_key=openai_api_key)

# -----------------------------
# Load embedding model (CPU only)
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Embedding model loaded successfully.")

# -----------------------------
# Answer generation function
# -----------------------------
def generate_answer(query, top_chunks):
    """
    Generate an answer to the query using the retrieved text chunks.
    """
    context_text = "\n\n".join([chunk["text"] for chunk in top_chunks])
    prompt = (
        f"Answer the question using the context below:\n\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # OpenAI v1 client call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content
    return answer

# -----------------------------
# Test the pipeline
# -----------------------------
if __name__ == "__main__":
    query = "What are the main cybersecurity threats?"
    
    print("Retrieving top chunks...")
    top_chunks = retrieve(query, top_k=5)  # Use your FAISS retriever

    print("Generating answer...")
    answer = generate_answer(query, top_chunks)

    print("\n=== ANSWER ===\n")
    print(answer)