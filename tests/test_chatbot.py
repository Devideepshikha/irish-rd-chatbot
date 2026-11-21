"""Unit tests for the parts of the chatbot that don't need the Gemini API.

Run with: pytest
"""
from ingest import chunk_text
from rag import search_stats


def test_chunks_have_the_right_size_and_overlap():
    text = "".join(str(i % 10) for i in range(2500))
    chunks = chunk_text(text, size=1000, overlap=200)
    assert [len(c) for c in chunks] == [1000, 1000, 900, 100]
    assert chunks[1][:200] == chunks[0][-200:]  # neighbours share 200 characters


def test_chunk_text_removes_line_breaks():
    assert chunk_text("R&D\n\nspending   grew") == ["R&D spending grew"]


def test_chunk_text_handles_empty_page():
    assert chunk_text("") == []


def write_stats(tmp_path):
    csv = tmp_path / "stats.csv"
    csv.write_text(
        "year,sector,measure,value,unit,source\n"
        "2023,business,BERD,100,EUR m,demo\n"
        "2022,higher education,HERD,50,EUR m,demo\n"
    )
    return csv


def test_search_returns_matching_row_first(tmp_path):
    result = search_stats("business 2023", csv_path=write_stats(tmp_path))
    rows = result.strip().splitlines()
    assert rows[0].startswith("year,sector")
    assert rows[1].startswith("2023,business")


def test_search_no_match(tmp_path):
    assert search_stats("agriculture", csv_path=write_stats(tmp_path)) == "No matching rows."


def test_search_missing_file(tmp_path):
    assert "missing" in search_stats("business", csv_path=tmp_path / "nope.csv")