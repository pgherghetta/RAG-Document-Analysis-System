import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Add parent folder to sys.path.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retriever import retrieve

# LLM Model Settings
LLM_MODEL_NAME = "google/flan-t5-large"
MAX_CHUNK_TOKENS = 200
TOP_K_CHUNKS = 3

print(f"Loading LLM model ({LLM_MODEL_NAME})...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

# Generate answer.
def generate_answer(query, top_chunks):

    context_chunks = []

    for chunk in top_chunks:
        tokens = tokenizer.encode(
            chunk["text"],
            truncation=True,
            max_length=MAX_CHUNK_TOKENS
        )

        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

        context_chunks.append(
            f"[Source: {chunk['source_file']}, Chunk {chunk['chunk_index']}]\n{truncated_text}"
        )

    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a research assistant answering questions about United States congress documents.

Use ONLY the information in the context to answer the question.
Write a complete answer in 3-5 sentences.
If possible, mention multiple relevant details from the context.

Context:
{context_text}

Question: {query}

Answer:
"""

    output = generator(prompt)

    return output[0]["generated_text"]

# Local test
if __name__ == "__main__":

    query = "What are current cybersecurity threats?"

    print("Retrieving top chunks...")
    top_chunks = retrieve(query, top_k=TOP_K_CHUNKS)

    print("Generating answer...")
    answer = generate_answer(query, top_chunks)

    print("\n=== ANSWER ===\n")
    print(answer)