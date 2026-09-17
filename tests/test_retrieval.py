import pytest

from documents import DATA_DIR, Chunk, load_chunks
from retrieval import (
    BM25Retriever,
    HybridRetriever,
    QueryFilters,
    apply_filters,
    extract_companies,
    extract_years,
    parse_filters,
    tokenize,
)

COMPANIES = ["Northstar Industrial", "BluePeak Software", "Harborlight Logistics"]


@pytest.fixture(scope="module")
def corpus():
    return load_chunks(DATA_DIR)


@pytest.fixture(scope="module")
def bm25(corpus):
    return BM25Retriever(corpus)


def test_tokenizer_keeps_numbers_and_decimals():
    assert tokenize("Margin fell to 17.0% from 20.0%") == [
        "margin",
        "fell",
        "to",
        "17.0",
        "from",
        "20.0",
    ]


def test_extract_years_returns_every_year_in_order():
    assert extract_years("change from 2023 to 2024") == ("2023", "2024")


def test_extract_years_deduplicates():
    assert extract_years("2024 versus 2024") == ("2024",)


def test_extract_years_ignores_non_years():
    assert extract_years("revenue of 1450 million") == ()


def test_extract_companies_matches_full_and_short_names():
    assert extract_companies("Why did Northstar's margin fall?", COMPANIES) == (
        "Northstar Industrial",
    )
    assert extract_companies("BluePeak Software revenue", COMPANIES) == ("BluePeak Software",)


def test_extract_companies_returns_all_mentioned():
    matched = extract_companies("Compare Northstar and BluePeak in 2024", COMPANIES)

    assert set(matched) == {"Northstar Industrial", "BluePeak Software"}


def test_extract_companies_respects_word_boundaries():
    assert extract_companies("the ship harborlights were visible", COMPANIES) == ()


def test_extract_companies_ignores_unknown_names():
    assert extract_companies("What was Acme's revenue in 2024?", COMPANIES) == ()


def sample_chunks():
    return [
        Chunk("Northstar Industrial", "2023", "16", "margin was 20.0 percent", "n23.txt"),
        Chunk("Northstar Industrial", "2024", "18", "margin declined to 17.0 percent", "n24.txt"),
        Chunk("BluePeak Software", "2024", "14", "margin improved to 21.9 percent", "b24.txt"),
    ]


def test_cross_year_question_keeps_both_years():
    chunks = sample_chunks()
    result = apply_filters(chunks, parse_filters("Northstar margin from 2023 to 2024", COMPANIES))

    years = {chunks[i].year for i in result.candidate_ids}

    assert years == {"2023", "2024"}
    assert result.relaxed is False


def test_comparison_question_keeps_both_companies():
    chunks = sample_chunks()
    result = apply_filters(chunks, parse_filters("Compare Northstar and BluePeak in 2024", COMPANIES))

    companies = {chunks[i].company for i in result.candidate_ids}

    assert companies == {"Northstar Industrial", "BluePeak Software"}
    assert all(chunks[i].year == "2024" for i in result.candidate_ids)


def test_unknown_company_does_not_filter_by_company():
    chunks = sample_chunks()
    result = apply_filters(chunks, parse_filters("What was Acme's margin in 2024?", COMPANIES))

    assert {chunks[i].year for i in result.candidate_ids} == {"2024"}
    assert len(result.candidate_ids) == 2


def test_impossible_filter_is_relaxed_instead_of_returning_nothing():
    chunks = sample_chunks()
    result = apply_filters(chunks, parse_filters("Northstar revenue in 2026", COMPANIES))

    assert result.relaxed is True
    assert result.candidate_ids
    # Relaxation drops the year but keeps the company, which is the more useful constraint.
    assert {chunks[i].company for i in result.candidate_ids} == {"Northstar Industrial"}
    assert result.applied == QueryFilters(companies=("Northstar Industrial",))


def test_year_only_filter_is_relaxed_to_unfiltered():
    chunks = sample_chunks()
    result = apply_filters(chunks, QueryFilters(years=("2019",)))

    assert result.relaxed is True
    assert len(result.candidate_ids) == len(chunks)


def test_describe_reports_the_relaxation():
    chunks = sample_chunks()
    result = apply_filters(chunks, parse_filters("Northstar revenue in 2026", COMPANIES))

    assert "relaxed from" in result.describe()


def test_retrieve_respects_the_metadata_filter(bm25):
    results = bm25.retrieve("Why did Northstar's EBITDA margin fall in 2024?", top_k=4)

    assert results
    assert {r.chunk.company for r in results} == {"Northstar Industrial"}
    assert {r.chunk.year for r in results} == {"2024"}


def test_retrieve_honours_top_k(bm25):
    assert len(bm25.retrieve("EBITDA margin", top_k=3)) == 3


def test_ranks_are_sequential(bm25):
    results = bm25.retrieve("EBITDA margin", top_k=5)

    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_scores_are_descending(bm25):
    scores = [r.score for r in bm25.retrieve("freight and energy costs", top_k=5)]

    assert scores == sorted(scores, reverse=True)


def test_bm25_ranks_the_margin_page_first_when_wording_matches(bm25):
    results = bm25.retrieve("Northstar 2024 adjusted EBITDA margin declined freight energy", top_k=4)

    assert results[0].chunk.page == "18"


def test_bm25_is_sensitive_to_vocabulary_mismatch(bm25):
    """Documents the weakness the hybrid retriever exists to cover.

    'fall' never appears on the page; the document says 'declined'. BM25 has no stemming
    and no synonyms, so the target page loses ground to pages that merely repeat the
    query's other words.
    """
    matched = bm25.retrieve("Northstar 2024 adjusted EBITDA margin declined", top_k=1)
    mismatched = bm25.retrieve("Northstar 2024 EBITDA margin fall", top_k=1)

    assert matched[0].chunk.page == "18"
    assert mismatched[0].chunk.page != "18"


def test_bm25_zero_score_for_unrelated_query(bm25):
    scores = bm25.score_all("xyzzy nonexistent token")

    assert max(scores) == 0.0


def test_hybrid_fusion_returns_results(corpus, bm25):
    hybrid = HybridRetriever([bm25, BM25Retriever(corpus, k1=1.2, b=0.6)])
    results = hybrid.retrieve("Why did BluePeak's EBITDA margin improve in 2024?", top_k=3)

    assert len(results) == 3
    assert {r.chunk.company for r in results} == {"BluePeak Software"}
    assert hybrid.name == "hybrid(bm25+bm25)"


def test_hybrid_requires_at_least_one_retriever():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_dedupe_pages_returns_distinct_pages(bm25):
    results = bm25.retrieve("Northstar 2024 freight energy costs", top_k=4, dedupe_pages=True)
    keys = [r.chunk.source_key for r in results]

    assert len(keys) == len(set(keys))


def test_dedupe_pages_keeps_the_best_chunk_of_a_page(bm25):
    query = "Northstar 2024 adjusted EBITDA margin declined freight energy"

    plain = bm25.retrieve(query, top_k=4)
    deduped = bm25.retrieve(query, top_k=4, dedupe_pages=True)

    assert deduped[0].chunk.text == plain[0].chunk.text
    assert deduped[0].score == plain[0].score
