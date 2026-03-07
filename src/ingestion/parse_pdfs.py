import os
from PyPDF2 import PdfReader

# -----------------------------
# Folders
# -----------------------------
RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# -----------------------------
# Get all PDF files
# -----------------------------
pdf_files = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs to process.")

# -----------------------------
# Extract text from each PDF
# -----------------------------
for pdf_file in pdf_files:
    pdf_path = os.path.join(RAW_FOLDER, pdf_file)
    txt_file_name = os.path.splitext(pdf_file)[0] + ".txt"
    txt_path = os.path.join(PROCESSED_FOLDER, txt_file_name)

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Optional: clean extra whitespace
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

        # Save to text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Processed {pdf_file} → {txt_file_name}")

    except Exception as e:
        print(f"Failed to process {pdf_file}: {e}")

print("PDF text extraction complete.")