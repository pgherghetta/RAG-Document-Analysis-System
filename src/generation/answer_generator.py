import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# -----------------------------
# Add parent folder to sys.path
# -----------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retriever import retrieve

# -----------------------------
# LLM Model Settings
# -----------------------------
LLM_MODEL_NAME = "google/flan-t5-large"
MAX_CHUNK_TOKENS = 100   # Truncate each chunk to avoid model limits.
TOP_K_CHUNKS = 3         # Number of chunks to use.

print(f"Loading LLM model ({LLM_MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

# Use a text2text-generation pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

# -----------------------------
# Embeddings model for retrieval
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_text(text):
    """Return embeddings for a single text string."""
    return embedding_model.encode(text)

# -----------------------------
# Generate answer using chunks directly
# -----------------------------
def generate_answer(query, top_chunks):
    """
    Generate an answer using retrieved document chunks.
    This version uses the full chunk text without summarization.
    """

    context_chunks = []
    for chunk in top_chunks:
        # Truncate chunk to avoid model limit
        tokens = tokenizer.encode(chunk["text"], truncation=True, max_length=MAX_CHUNK_TOKENS)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        context_chunks.append(f"[Source: {chunk['source_file']}, Chunk {chunk['chunk_index']}]\n{truncated_text}")

    context_text = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the information below.
Do NOT make up information. Be concise (3-5 sentences). Cite sources using (Source: filename, Chunk number).

Context:
{context_text}

Question:
{query}

Answer:
"""
    output = generator(prompt)
    return output[0]["generated_text"]

# -----------------------------
# Local test
# -----------------------------
if __name__ == "__main__":
    query = "What are some current strategic allies of the United States?"
    print("Retrieving top chunks...")
    top_chunks = retrieve(query, top_k=TOP_K_CHUNKS)
    print("Generating answer...")
    answer = generate_answer(query, top_chunks)
    print("\n=== ANSWER ===\n")
    print(answer)