"""Retrieval: metadata filtering, a BM25 baseline, dense embeddings, and RRF hybrid.

The filtering and BM25 code is pure standard library so it can be tested without numpy or
a downloaded model. numpy and sentence-transformers are imported lazily inside
DenseRetriever.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from documents import Chunk

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used by BM25.

    No stemming: 'declined' and 'decline' are different terms. That is a real weakness of
    the lexical baseline and part of why the dense retriever is worth its cost.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class QueryFilters:
    companies: Tuple[str, ...] = ()
    years: Tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.companies and not self.years

    def describe(self) -> str:
        if self.is_empty:
            return "none"
        parts = []
        if self.companies:
            parts.append("company in {" + ", ".join(self.companies) + "}")
        if self.years:
            parts.append("year in {" + ", ".join(self.years) + "}")
        return " AND ".join(parts)


@dataclass
class FilterResult:
    candidate_ids: List[int]
    filters: QueryFilters
    applied: QueryFilters
    relaxed: bool = False

    def describe(self) -> str:
        if not self.relaxed:
            return self.applied.describe()
        return f"{self.applied.describe()} (relaxed from: {self.filters.describe()})"


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int


def extract_years(query: str) -> Tuple[str, ...]:
    """All four-digit years in the query, in order of appearance, de-duplicated.

    Taking every year (not just the first) is what makes 'from 2023 to 2024' work.
    """
    seen: List[str] = []
    for match in _YEAR_RE.finditer(query):
        year = match.group(0)
        if year not in seen:
            seen.append(year)
    return tuple(seen)


def extract_companies(query: str, known_companies: Iterable[str]) -> Tuple[str, ...]:
    """All known companies mentioned in the query, by full name or first word.

    Matching every company (not just the first) is what makes comparison questions work.
    An unrecognised company name simply yields no match, which leaves the pool unfiltered
    rather than silently filtering to the wrong issuer.
    """
    query_lower = query.lower()
    matched: List[str] = []

    for company in sorted(set(known_companies), key=len, reverse=True):
        name = company.lower()
        short_name = company.split()[0].lower()

        if _contains_word(query_lower, name) or _contains_word(query_lower, short_name):
            if company not in matched:
                matched.append(company)

    return tuple(matched)


def _contains_word(haystack: str, needle: str) -> bool:
    """Substring match on word boundaries, so 'peak' does not match 'peaked'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def parse_filters(query: str, known_companies: Iterable[str]) -> QueryFilters:
    return QueryFilters(
        companies=extract_companies(query, known_companies),
        years=extract_years(query),
    )


def apply_filters(chunks: Sequence[Chunk], filters: QueryFilters) -> FilterResult:
    """Apply metadata filters with graceful degradation.

    Order of attempts: company AND year -> company only -> year only -> unfiltered.
    A hard filter that empties the pool is worse than no filter, because the generator
    then receives no evidence at all and cannot even attempt a grounded answer.
    """
    attempts = [filters]

    if filters.companies and filters.years:
        attempts.append(QueryFilters(companies=filters.companies))
        attempts.append(QueryFilters(years=filters.years))

    attempts.append(QueryFilters())

    for attempt in attempts:
        ids = [
            i
            for i, chunk in enumerate(chunks)
            if (not attempt.companies or chunk.company in attempt.companies)
            and (not attempt.years or chunk.year in attempt.years)
        ]

        if ids:
            return FilterResult(
                candidate_ids=ids,
                filters=filters,
                applied=attempt,
                relaxed=attempt != filters,
            )

    return FilterResult(candidate_ids=[], filters=filters, applied=QueryFilters(), relaxed=True)


class BaseRetriever:
    name = "base"

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.companies = sorted({chunk.company for chunk in self.chunks})

    def filter_for(self, query: str) -> FilterResult:
        return apply_filters(self.chunks, parse_filters(query, self.companies))

    def score_all(self, query: str) -> List[float]:
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        dedupe_pages: bool = False,
    ) -> List[RetrievalResult]:
        """Return the top-k chunks that survive metadata filtering.

        With ``dedupe_pages`` only the best-scoring chunk per source page is kept. Chunk
        overlap means two adjacent chunks of the same page often score almost identically,
        which can fill the top-k with near-duplicate text and crowd out a second document
        that the question actually needs.
        """
        result = self.filter_for(query)

        if not result.candidate_ids:
            return []

        scores = self.score_all(query)

        ordered = sorted(result.candidate_ids, key=lambda i: (-scores[i], i))

        if dedupe_pages:
            seen_pages = set()
            deduped = []
            for i in ordered:
                key = self.chunks[i].source_key
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                deduped.append(i)
            ordered = deduped

        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ordered[:top_k], start=1)
        ]


class BM25Retriever(BaseRetriever):
    """Okapi BM25, implemented directly so the lexical baseline has no hidden magic."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.k1 = k1
        self.b = b

        self.doc_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]

        document_frequency: Counter = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))

        n_docs = len(self.chunks)
        self.idf: Dict[str, float] = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def score_all(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        scores = [0.0] * len(self.chunks)

        for i, term_freq in enumerate(self.term_frequencies):
            length_norm = self.k1 * (
                1 - self.b + self.b * (self.doc_lengths[i] / (self.avg_doc_length or 1))
            )
            total = 0.0

            for term in query_terms:
                freq = term_freq.get(term)
                if not freq:
                    continue
                total += self.idf[term] * (freq * (self.k1 + 1)) / (freq + length_norm)

            scores[i] = total

        return scores


class DenseRetriever(BaseRetriever):
    """Sentence-embedding retrieval with cosine similarity.

    Embeddings are L2-normalised at encode time, so a dot product is the cosine
    similarity and no per-query normalisation is needed.
    """

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], model_name: str = DEFAULT_EMBEDDING_MODEL):
        super().__init__(chunks)

        # Imported here so that importing this module (and running the lexical tests)
        # does not require torch to be installed.
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def score_all(self, query: str) -> List[float]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return self._np.dot(self.embeddings, query_embedding).tolist()


class HybridRetriever(BaseRetriever):
    """Reciprocal rank fusion of two retrievers.

    RRF combines rankings rather than scores, which avoids having to normalise a BM25
    score and a cosine similarity onto a common scale.
    """

    name = "hybrid"

    def __init__(self, retrievers: Sequence[BaseRetriever], k: int = 60):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")

        super().__init__(retrievers[0].chunks)
        self.retrievers = list(retrievers)
        self.k = k
        self.name = "hybrid(" + "+".join(r.name for r in self.retrievers) + ")"

    def score_all(self, query: str) -> List[float]:
        fused = [0.0] * len(self.chunks)

        for retriever in self.retrievers:
            scores = retriever.score_all(query)
            order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
            for rank, i in enumerate(order, start=1):
                fused[i] += 1.0 / (self.k + rank)

        return fused


def build_retriever(
    kind: str,
    chunks: Sequence[Chunk],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> BaseRetriever:
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "dense":
        return DenseRetriever(chunks, embedding_model)
    if kind == "hybrid":
        return HybridRetriever([DenseRetriever(chunks, embedding_model), BM25Retriever(chunks)])
    raise ValueError(f"unknown retriever kind: {kind!r} (expected bm25, dense or hybrid)")
