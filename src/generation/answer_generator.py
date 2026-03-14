import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# -----------------------------
# Add parent folder to sys.path.
# -----------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retriever import retrieve

# -----------------------------
# LLM Model Settings
# -----------------------------
LLM_MODEL_NAME = "google/flan-t5-large"
MAX_CHUNK_TOKENS = 150   # Truncate each chunk to avoid model limits.
TOP_K_CHUNKS = 3         # Number of chunks to use.

print(f"Loading LLM model ({LLM_MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

# Use a text2text-generation pipeline.
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
# Summarize each chunk individually.
# -----------------------------
def summarize_chunk(chunk_text, max_tokens=100):
    prompt = f"""
    Summarize the following text in 2-3 concise sentences, keeping all facts intact. Do not add extra information.

    Text:
    {chunk_text}

    Summary:
    """
    output = generator(prompt, max_new_tokens=max_tokens)
    return output[0]["generated_text"]

# -----------------------------
# Generate answer using summarized chunks.
# -----------------------------
def generate_answer(query, top_chunks):
    """
    Generate an answer using retrieved document chunks.
    Each chunk is first summarized to reduce input length and improve fluency.
    """

    summarized_chunks = []
    for chunk in top_chunks:
        # Truncate and summarize chunk.
        tokens = tokenizer.encode(chunk["text"], truncation=True, max_length=MAX_CHUNK_TOKENS)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        summary = summarize_chunk(truncated_text)
        summarized_chunks.append(f"[Source: {chunk['source_file']}, Chunk {chunk['chunk_index']}]\n{summary}")

    context_text = "\n\n".join(summarized_chunks)

    prompt = f"""
    Answer the question using ONLY the summarized information below.
    Do NOT make up information. Be concise (3-5 sentences). Cite sources using (Source: filename, Chunk number).

    Summarized Context:
    {context_text}

    Question:
    {query}

    Answer:
    """
    output = generator(prompt)
    return output[0]["generated_text"]




if __name__ == "__main__":
    query = "What are the main cybersecurity threats?"
    print("Retrieving top chunks...")
    top_chunks = retrieve(query, top_k=TOP_K_CHUNKS)

    print("Generating answer with summarized chunks...")
    answer = generate_answer(query, top_chunks)

    print("\n=== ANSWER ===\n")
    print(answer)