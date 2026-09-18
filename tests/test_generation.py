import pytest

from documents import Chunk
from generation import (
    NO_EVIDENCE_ANSWER,
    TokenUsage,
    build_prompt,
    extract_citations,
    generate_answer,
    validate_citations,
)
from retrieval import RetrievalResult

CHUNKS = [
    Chunk("Northstar Industrial", "2024", "18", "Margin declined to 17.0 percent.", "n24.txt"),
    Chunk("BluePeak Software", "2024", "14", "Margin improved to 21.9 percent.", "b24.txt"),
]

RETRIEVED = [
    RetrievalResult(chunk=CHUNKS[0], score=0.71, rank=1),
    RetrievalResult(chunk=CHUNKS[1], score=0.44, rank=2),
]


def test_prompt_contains_the_context_and_its_citations():
    prompt = build_prompt("Why did the margin fall?", RETRIEVED)

    assert "[Northstar Industrial 2024, p.18]" in prompt
    assert "Margin declined to 17.0 percent." in prompt
    assert "Why did the margin fall?" in prompt
    assert "INSUFFICIENT_EVIDENCE" in prompt


def test_extract_citations_normalises_and_deduplicates():
    answer = (
        "Margin fell [Northstar Industrial 2024, p. 18] because of freight "
        "[Northstar Industrial 2024, p.18]."
    )

    assert extract_citations(answer) == ["[Northstar Industrial 2024, p.18]"]


def test_extract_citations_ignores_other_bracketed_text():
    assert extract_citations("see note [6] and table [A]") == []


def test_validate_citations_accepts_supported_citations():
    report = validate_citations("Margin fell [Northstar Industrial 2024, p.18].", RETRIEVED)

    assert report.is_valid
    assert report.supported == ["[Northstar Industrial 2024, p.18]"]
    assert report.unsupported == []
    assert report.unused_sources == ["[BluePeak Software 2024, p.14]"]


def test_validate_citations_flags_a_page_that_was_never_retrieved():
    report = validate_citations("Margin fell [Northstar Industrial 2024, p.19].", RETRIEVED)

    assert not report.is_valid
    assert report.unsupported == ["[Northstar Industrial 2024, p.19]"]


def test_validate_citations_flags_a_year_that_was_never_retrieved():
    report = validate_citations("Margin was 20 percent [Northstar Industrial 2023, p.16].", RETRIEVED)

    assert report.unsupported == ["[Northstar Industrial 2023, p.16]"]


def test_answer_without_citations_is_detected():
    report = validate_citations("The margin fell because costs rose.", RETRIEVED)

    assert report.has_citations is False
    assert report.is_valid  # nothing unsupported, but nothing cited either


def test_no_retrieved_context_means_no_llm_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "some-model")

    result = generate_answer("What is the 2026 revenue?", [])

    assert result.called_llm is False
    assert result.refused is True
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.usage.total_tokens == 0


def test_missing_credentials_raise_only_when_a_call_is_needed(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        generate_answer("Why did the margin fall?", RETRIEVED)


def test_token_usage_cost():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)

    assert usage.total_tokens == 1_500_000
    assert usage.cost_usd(0.15, 0.60) == pytest.approx(0.45)
    assert usage.cost_usd(None, 0.60) is None


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, text, input_tokens=120, output_tokens=40):
        self.output_text = text
        self.usage = _Usage(input_tokens, output_tokens)


class _StubClient:
    """Minimal stand-in for openai.OpenAI, recording the calls it receives."""

    def __init__(self, text, reject_temperature=False):
        self.text = text
        self.reject_temperature = reject_temperature
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_temperature and "temperature" in kwargs:
            raise ValueError("Unsupported parameter: 'temperature' is not supported")
        return _Response(self.text)


def test_generate_answer_reports_usage_and_validates_citations(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "stub-model")
    client = _StubClient("Margin fell to 17.0% [Northstar Industrial 2024, p.18].")

    result = generate_answer("Why did the margin fall?", RETRIEVED, client=client)

    assert result.called_llm is True
    assert result.refused is False
    assert result.usage.total_tokens == 160
    assert result.citations.is_valid
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["model"] == "stub-model"
    assert client.calls[0]["instructions"]
    assert client.calls[0]["store"] is False


def test_generate_answer_retries_without_temperature_when_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "reasoning-model")
    client = _StubClient("Answer [BluePeak Software 2024, p.14].", reject_temperature=True)

    result = generate_answer("Why did the margin improve?", RETRIEVED, client=client)

    assert len(client.calls) == 2
    assert "temperature" in client.calls[0]
    assert "temperature" not in client.calls[1]
    assert result.citations.supported == ["[BluePeak Software 2024, p.14]"]


def test_generate_answer_detects_refusal(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "stub-model")
    client = _StubClient("INSUFFICIENT_EVIDENCE the context does not give a 2026 figure.")

    result = generate_answer("What was revenue in 2026?", RETRIEVED, client=client)

    assert result.refused is True
    assert result.citations.has_citations is False


def test_generate_answer_flags_a_hallucinated_citation(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "stub-model")
    client = _StubClient("Revenue grew [Northstar Industrial 2024, p.12].")

    result = generate_answer("What was revenue?", RETRIEVED, client=client)

    assert result.citations.unsupported == ["[Northstar Industrial 2024, p.12]"]
    assert not result.citations.is_valid


def test_unexpected_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "stub-model")

    class _Boom:
        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        generate_answer("Why?", RETRIEVED, client=_Boom())
