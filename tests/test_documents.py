from pathlib import Path

import pytest

from documents import (
    DATA_DIR,
    Chunk,
    DocumentFormatError,
    chunk_text,
    corpus_stats,
    load_chunks,
    parse_document,
)


def test_short_text_is_one_chunk():
    assert chunk_text("one two three", max_words=90) == ["one two three"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ") == []


def test_long_text_is_split_with_overlap():
    words = [f"w{i}" for i in range(200)]
    chunks = chunk_text(" ".join(words), max_words=90, overlap_words=15)

    assert len(chunks) == 3

    first = chunks[0].split()
    second = chunks[1].split()

    assert len(first) == 90
    # The overlap is the tail of the previous chunk repeated at the head of the next one.
    assert second[:15] == first[-15:]


def test_every_word_survives_chunking():
    words = [f"w{i}" for i in range(413)]
    max_words, overlap = 50, 10
    chunks = chunk_text(" ".join(words), max_words=max_words, overlap_words=overlap)

    # Rebuild the original by dropping the overlapping head of every chunk after the first.
    recovered = chunks[0].split()
    for chunk in chunks[1:]:
        recovered.extend(chunk.split()[overlap:])

    assert recovered == words
    assert all(len(chunk.split()) <= max_words for chunk in chunks)


def test_final_chunk_is_not_dropped():
    words = [f"w{i}" for i in range(95)]
    chunks = chunk_text(" ".join(words), max_words=90, overlap_words=15)

    assert chunks[-1].split()[-1] == "w94"


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_chunk_parameters_are_rejected(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=max_words, overlap_words=overlap)


def test_parse_document_extracts_metadata_and_pages(tmp_path: Path):
    path = tmp_path / "demo_2024.txt"
    path.write_text(
        "COMPANY: Demo Corp\nYEAR: 2024\n\n[PAGE 7]\nfirst page\n\n[PAGE 9]\nsecond page\n",
        encoding="utf-8",
    )

    chunks = parse_document(path)

    assert [c.page for c in chunks] == ["7", "9"]
    assert all(c.company == "Demo Corp" and c.year == "2024" for c in chunks)
    assert chunks[0].source_file == "demo_2024.txt"
    assert chunks[0].text == "first page"


@pytest.mark.parametrize(
    "content",
    [
        "YEAR: 2024\n\n[PAGE 1]\ntext",
        "COMPANY: Demo Corp\n\n[PAGE 1]\ntext",
        "COMPANY: Demo Corp\nYEAR: 2024\n\nno page markers",
    ],
)
def test_malformed_documents_raise_a_clear_error(tmp_path: Path, content: str):
    path = tmp_path / "broken.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DocumentFormatError):
        parse_document(path)


def test_missing_data_directory_raises(tmp_path: Path):
    with pytest.raises(DocumentFormatError):
        load_chunks(tmp_path)


def test_citation_format():
    chunk = Chunk("Northstar Industrial", "2024", "18", "text", "northstar_2024.txt")

    assert chunk.citation == "[Northstar Industrial 2024, p.18]"
    assert chunk.source_key == ("Northstar Industrial", "2024", "18")


def test_page_numbers_are_preserved_for_multi_chunk_pages(tmp_path: Path):
    long_page = " ".join(f"w{i}" for i in range(250))
    path = tmp_path / "demo_2024.txt"
    path.write_text(f"COMPANY: Demo Corp\nYEAR: 2024\n\n[PAGE 5]\n{long_page}\n", encoding="utf-8")

    chunks = parse_document(path)

    assert len(chunks) > 1
    assert {c.page for c in chunks} == {"5"}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_real_corpus_actually_exercises_the_chunker():
    """Regression test for the first review: the corpus used to be too small to split."""
    stats = corpus_stats(load_chunks(DATA_DIR))

    assert stats["documents"] >= 4
    assert stats["pages"] >= 30
    assert stats["chunks"] > stats["pages"]
    assert stats["split_pages"] > 0
