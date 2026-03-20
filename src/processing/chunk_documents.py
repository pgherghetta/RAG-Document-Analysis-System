import os
import json

# Folders
PROCESSED_FOLDER = "data/processed"    # Input text files
CHUNKS_FOLDER = os.path.join(PROCESSED_FOLDER, "chunks")  # Output folder
os.makedirs(CHUNKS_FOLDER, exist_ok=True)

# Chunking parameters.
WORDS_PER_CHUNK = 500  
# Number of words to overlap between chunks for context.
OVERLAP = 50         

# Get all text files.
text_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.lower().endswith(".txt")]
print(f"Found {len(text_files)} text files to chunk.")

all_chunks_metadata = []

# Process each text file.
for txt_file in text_files:
    txt_path = os.path.join(PROCESSED_FOLDER, txt_file)
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    start = 0
    chunk_index = 1

    while start < len(words):
        end = start + WORDS_PER_CHUNK
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunk_file_name = f"{os.path.splitext(txt_file)[0]}_chunk{chunk_index:03d}.txt"
        chunk_path = os.path.join(CHUNKS_FOLDER, chunk_file_name)

        # Save chunk.
        with open(chunk_path, "w", encoding="utf-8") as cf:
            cf.write(chunk_text)

        # Save metadata for RAG (source file + chunk number).
        all_chunks_metadata.append({
            "chunk_file": chunk_file_name,
            "source_file": txt_file,
            "chunk_index": chunk_index,
            "num_words": len(chunk_words)
        })

        # Move start forward, keeping overlap.
        start += WORDS_PER_CHUNK - OVERLAP
        chunk_index += 1

print(f"Total chunks created: {len(all_chunks_metadata)}")

# Save metadata JSON.
metadata_path = os.path.join(CHUNKS_FOLDER, "chunks_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks_metadata, f, indent=4)

print(f"Chunk metadata saved to {metadata_path}")