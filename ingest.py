"""Step 1: build the search index from the PDF reports in data/reports.

Reads each PDF page by page, splits the text into overlapping chunks,
turns every chunk into an embedding with the OpenAI API and saves the
result to the index/ folder. Run this once, and again whenever you add
or change a report.
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

REPORTS_DIR = Path("data/reports")
INDEX_DIR = Path("index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 1000  # characters per chunk
OVERLAP = 200      # characters shared between neighbouring chunks
MIN_CHUNK = 100    # skip tiny fragments such as page numbers


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping pieces so no sentence is lost at a boundary."""
    text = " ".join(text.split())  # collapse line breaks and extra spaces
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def load_chunks():
    """Return a list of {source, page, text} for every chunk in every PDF."""
    chunks = []
    for pdf_path in sorted(REPORTS_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for piece in chunk_text(text):
                if len(piece) >= MIN_CHUNK:
                    chunks.append({"source": pdf_path.name, "page": page_number, "text": piece})
    return chunks


def embed(client, texts, batch_size=100):
    """Embed texts in batches and return a 2-D numpy array."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(item.embedding for item in response.data)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    return np.array(vectors, dtype=np.float32)


def main():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No text found. Put one or more PDF reports in data/reports/ first.")
    reports = sorted({c["source"] for c in chunks})
    print(f"Found {len(chunks)} chunks in {len(reports)} reports: {', '.join(reports)}")

    vectors = embed(OpenAI(), [c["text"] for c in chunks])

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    print("Index saved to index/. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()