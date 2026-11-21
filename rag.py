"""Step 2: answer questions with retrieval-augmented generation (RAG).

For each question:
1. Retrieve the report chunks closest in meaning to the question.
2. Send them to the Gemini model with the question.
3. The model can call one tool, lookup_rd_stats, when it needs an exact
   figure. This is the agent step: the model decides for itself whether
   to use the tool, and the Gemini SDK runs the tool and passes the
   result back to the model.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
INDEX_DIR = Path("index")
STATS_CSV = Path("data/rd_stats.csv")
TOP_K = 4

SYSTEM_PROMPT = (
    "You answer questions about innovation and R&D in Ireland. "
    "Use only the report excerpts you are given and the lookup_rd_stats tool. "
    "Cite every fact taken from a report as [file name, p. N]. "
    "When a question needs an exact figure, call lookup_rd_stats and say the number "
    "came from the R&D statistics table. "
    "If the answer is not in the excerpts or the table, say you don't know."
)

_client = None
_index = None
_tool_calls = []  # queries the model sent to the tool during the current answer


def get_client():
    """Create the Gemini client on first use (keeps the module importable in tests)."""
    global _client
    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY from .env
    return _client


def load_index():
    """Load the chunks and their normalised embeddings once, on first use."""
    global _index
    if _index is None:
        chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(INDEX_DIR / "embeddings.npy")
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        _index = (chunks, vectors)
    return _index


def retrieve(question, k=TOP_K):
    """Return the k chunks most similar in meaning to the question."""
    chunks, vectors = load_index()
    response = get_client().models.embed_content(model=EMBED_MODEL, contents=[question])
    q = np.array(response.embeddings[0].values, dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = vectors @ q  # cosine similarity, because both sides are normalised
    best = np.argsort(scores)[::-1][:k]
    return [dict(chunks[i], score=float(scores[i])) for i in best]


def search_stats(query, csv_path=STATS_CSV):
    """Return rows of the statistics table that match the query words, best first."""
    if not Path(csv_path).exists():
        return "The R&D statistics table is missing."
    table = pd.read_csv(csv_path)
    if table.empty:
        return "The R&D statistics table is empty."
    words = [w.lower() for w in query.split() if len(w) > 2]
    row_text = table.astype(str).apply(lambda row: " ".join(row).lower(), axis=1)
    matches = row_text.apply(lambda text: sum(word in text for word in words))
    hits = table[matches > 0].assign(_matches=matches[matches > 0])
    hits = hits.sort_values("_matches", ascending=False).drop(columns="_matches").head(10)
    return "No matching rows." if hits.empty else hits.to_csv(index=False)


def lookup_rd_stats(query: str) -> str:
    """Look up exact R&D figures, such as spending by year or sector, in the R&D statistics table.

    Use this when the question needs an exact number.

    Args:
        query: Keywords to search for, for example 'foreign-owned 2023'.
    """
    _tool_calls.append(query)
    return search_stats(query)


def generate_with_retry(retries=4, **request):
    """Call Gemini, waiting and retrying when the free tier is busy (429/500/503)."""
    for attempt in range(retries):
        try:
            return get_client().models.generate_content(**request)
        except (errors.ServerError, errors.ClientError) as error:
            if error.code in (429, 500, 503) and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"Gemini is busy ({error.code}), retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise


def answer(question):
    """Return (answer text, source chunks, whether the tool was used)."""
    sources = retrieve(question)
    excerpts = "\n\n".join(f"[{s['source']}, p. {s['page']}]\n{s['text']}" for s in sources)
    _tool_calls.clear()
    response = generate_with_retry(
        model=CHAT_MODEL,
        contents=f"Report excerpts:\n{excerpts}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[lookup_rd_stats],  # the SDK runs the tool when the model asks for it
        ),
    )
    return response.text, sources, bool(_tool_calls)