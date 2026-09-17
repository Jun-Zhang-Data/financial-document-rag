"""Document loading: parse source files into chunks that carry their own citation.

Pure standard library on purpose, so the data layer can be tested without installing
numpy or a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEFAULT_MAX_WORDS = 90
DEFAULT_OVERLAP_WORDS = 15

_COMPANY_RE = re.compile(r"^COMPANY:\s*(.+)$", re.MULTILINE)
_YEAR_RE = re.compile(r"^YEAR:\s*(\d{4})\s*$", re.MULTILINE)
_PAGE_SPLIT_RE = re.compile(r"\[PAGE\s+(\d+)\]")


class DocumentFormatError(ValueError):
    """Raised when a source document does not match the expected header format."""


@dataclass(frozen=True)
class Chunk:
    company: str
    year: str
    page: str
    text: str
    source_file: str
    chunk_index: int = 0

    @property
    def citation(self) -> str:
        return f"[{self.company} {self.year}, p.{self.page}]"

    @property
    def source_key(self) -> tuple[str, str, str]:
        """Identity used by the evaluation: which page of which report this came from."""
        return (self.company, self.year, self.page)


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[str]:
    """Split text into fixed-size word windows with overlap.

    Overlap exists so that a sentence sitting on a chunk boundary still appears whole in
    one of the two neighbouring chunks. The cost is duplicated text, which inflates the
    index and can put two near-identical chunks in the same top-k.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and < max_words")

    words = text.split()

    if len(words) <= max_words:
        return [" ".join(words)] if words else []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


def parse_document(
    path: Path,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[Chunk]:
    """Read one document, extract metadata, and create structured chunks."""
    raw = path.read_text(encoding="utf-8")

    company_match = _COMPANY_RE.search(raw)
    year_match = _YEAR_RE.search(raw)

    if company_match is None:
        raise DocumentFormatError(f"{path.name}: missing 'COMPANY: <name>' header line")
    if year_match is None:
        raise DocumentFormatError(f"{path.name}: missing 'YEAR: <yyyy>' header line")

    company = company_match.group(1).strip()
    year = year_match.group(1).strip()

    parts = _PAGE_SPLIT_RE.split(raw)

    if len(parts) < 3:
        raise DocumentFormatError(f"{path.name}: no '[PAGE n]' markers found")

    chunks: List[Chunk] = []

    # parts == [preamble, page_no, page_text, page_no, page_text, ...]
    for i in range(1, len(parts), 2):
        page = parts[i].strip()
        page_text = parts[i + 1].strip()

        for index, piece in enumerate(chunk_text(page_text, max_words, overlap_words)):
            chunks.append(
                Chunk(
                    company=company,
                    year=year,
                    page=page,
                    text=piece,
                    source_file=path.name,
                    chunk_index=index,
                )
            )

    return chunks


def load_chunks(data_dir: Path | None = None) -> List[Chunk]:
    """Load every document in the data directory and combine their chunks."""
    directory = data_dir or DATA_DIR
    paths = sorted(directory.glob("*.txt"))

    if not paths:
        raise DocumentFormatError(f"no .txt documents found in {directory}")

    chunks: List[Chunk] = []

    for path in paths:
        chunks.extend(parse_document(path))

    return chunks


def corpus_stats(chunks: List[Chunk]) -> dict:
    """Small summary used by the CLI and the evaluation report."""
    word_counts = [len(chunk.text.split()) for chunk in chunks]

    return {
        "documents": len({chunk.source_file for chunk in chunks}),
        "pages": len({chunk.source_key for chunk in chunks}),
        "chunks": len(chunks),
        "split_pages": len({c.source_key for c in chunks if c.chunk_index > 0}),
        "max_chunk_words": max(word_counts) if word_counts else 0,
        "mean_chunk_words": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0.0,
    }
