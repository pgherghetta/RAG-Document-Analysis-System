# 📄 Advanced RAG Document Analysis System

## Overview

This project is an **end-to-end Retrieval-Augmented Generation (RAG) system** designed to analyze and answer questions over a corpus of congressional reports from congress.gov. Instead of relying on general-purpose AI knowledge, the system retrieves relevant document chunks and generates grounded, source-backed answers. This was built as part of a hands-on exploration into applied AI/ML engineering and RAG systems.

---

## What This Project Demonstrates

* Building a **production-style RAG pipeline**
* Working with **vector databases (FAISS)** for semantic search
* Integrating **LLMs (FLAN-T5)** for grounded generation
* Designing **modular ML systems**
* Evaluating RAG performance with **retrieval + generation metrics**
* Deploying an interactive UI with **Streamlit**

---

## System Architecture

```
PDF Documents
      ↓
Text Extraction
      ↓
Chunking + Metadata
      ↓
Embeddings (Sentence Transformers)
      ↓
FAISS Vector Database
      ↓
Retriever (semantic search)
      ↓
LLM (FLAN-T5)
      ↓
Answer with Sources
      ↓
Evaluation Metrics
```

---

## Key Features

### Document Ingestion

* Parses PDFs into clean text
* Splits documents into semantically meaningful chunks
* Stores metadata (source file, chunk index)

### Semantic Retrieval

* Uses **Sentence Transformers** to embed text
* Stores vectors in a **FAISS index**
* Retrieves top-k relevant chunks for any query

### LLM Answer Generation

* Uses **FLAN-T5** for local inference (no API required)
* Generates answers grounded in retrieved context
* Includes **source attribution**

### Interactive UI

* Built with **Streamlit**
* Users can:

  * Ask questions
  * View generated answers
  * Expand/collapse retrieved document chunks

### Evaluation Pipeline

* Local evaluation (CPU-friendly)
* Metrics include:

  * **ROUGE** (text overlap)
  * **BERTScore** (semantic similarity)
  * **Custom retrieval overlap score**
* Highlights tradeoffs between:

  * Retrieval quality
  * Answer relevance
  * Semantic similarity vs factual accuracy

---

## Example Use Case

**Query:**

```
What are current cybersecurity threats?
```

**System Output:**

* Retrieves relevant document chunks
* Generates a concise answer
* Displays sources for verification

---

## Tech Stack

* **Python**
* **Transformers (Hugging Face)**
* **Sentence Transformers**
* **FAISS**
* **Streamlit**
* **Evaluate (ROUGE, BERTScore)**

---

## Project Structure

```
src/
├── ingestion/        # PDF parsing
├── processing/       # chunking + embeddings
├── retrieval/        # FAISS + retriever
├── generation/       # LLM answer generation
├── evaluation/       # RAG evaluation scripts
├── app/              # Streamlit UI
```

---

## How to Run

### 1. Setup environment

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the app

```
streamlit run src/app/streamlit_app.py
```

### 3. Run evaluation

```
python src/evaluation/local_rag_evaluation.py
```

---

