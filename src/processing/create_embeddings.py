import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
CHUNKS_FOLDER = "data/processed/chunks"
METADATA_FILE = os.path.join(CHUNKS_FOLDER, "chunks_metadata.json")

VECTOR_STORE_FOLDER = "data/vector_store"
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_FOLDER, "faiss_index.bin")
EMBEDDINGS_METADATA_PATH = os.path.join(VECTOR_STORE_FOLDER, "embeddings_metadata.json")

# Load chunk metadata.
print("Loading chunk metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")

# Load embedding model.
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare texts for embedding.
texts = []

for chunk in chunks:
    chunk_file = chunk["chunk_file"]
    chunk_path = os.path.join(CHUNKS_FOLDER, chunk_file)

    with open(chunk_path, "r", encoding="utf-8") as f:
        text = f.read()

    texts.append(text)

print(f"Creating embeddings for {len(texts)} chunks...")

# Create embeddings.
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# Create FAISS index.
dimension = embeddings.shape[1]

print(f"Embedding dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")

# Save FAISS index.
faiss.write_index(index, FAISS_INDEX_PATH)

print(f"FAISS index saved to {FAISS_INDEX_PATH}")

# Save metadata for retrieval.
with open(EMBEDDINGS_METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4)

print(f"Embedding metadata saved to {EMBEDDINGS_METADATA_PATH}")

print("Embedding pipeline completed successfully.")