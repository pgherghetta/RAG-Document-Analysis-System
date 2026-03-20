import os
import sys
import pandas as pd

# Prevent parallel tokenizer warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retriever import retrieve
from generation.answer_generator import generate_answer

# Text similarity metrics.
from evaluate import load
rouge = load("rouge")
bert_score = load("bertscore")

# Evaluation Questions & References
evaluation_data = [
    {
        "question": "What are current cybersecurity threats?",
        "reference_answer": (
            "Current cybersecurity threats include ransomware attacks, supply chain vulnerabilities, "
            "critical infrastructure targeting, and phishing campaigns. Organizations are also seeing "
            "increased attacks on cloud services and IoT devices."
        )
    },
]


# Run RAG pipeline.
records = []

print("\nRunning local RAG evaluation...\n")

for entry in evaluation_data:
    question = entry["question"]
    ref_answer = entry["reference_answer"]

    print(f"Processing: {question}")

    # Retrieve top chunks.
    chunks = retrieve(question, top_k=3)

    # Truncate chunk text to avoid tokenizer overflow.
    contexts = [chunk["text"][:600] for chunk in chunks]

    # Generate answer using local FLAN-T5.
    answer = generate_answer(question, chunks)

    # Compute ROUGE and BERTScore.
    rouge_score = rouge.compute(predictions=[answer], references=[ref_answer])
    bertscore_score = bert_score.compute(predictions=[answer], references=[ref_answer], lang="en")

    # Simple heuristic retrieval score: fraction of reference keywords found in chunks.
    keywords = set(ref_answer.lower().split())
    retrieved_text = " ".join([c.lower() for c in contexts])
    overlap = sum(1 for kw in keywords if kw in retrieved_text)
    retrieval_score = overlap / max(1, len(keywords))

    # Append record.
    records.append({
        "question": question,
        "reference_answer": ref_answer,
        "generated_answer": answer,
        "contexts": contexts,
        "retrieval_overlap_score": retrieval_score,
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "bert_precision": bertscore_score["precision"][0],
        "bert_recall": bertscore_score["recall"][0],
        "bert_f1": bertscore_score["f1"][0]
    })

# Save results.
df = pd.DataFrame(records)
output_path = "./local_rag_evaluation_simple.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved evaluation results to {output_path}")
print("\nEvaluation complete!")