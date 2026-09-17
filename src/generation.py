"""Generation: prompt construction, the LLM call, citation validation and cost reporting."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from retrieval import RetrievalResult

CITATION_RE = re.compile(r"\[([^\[\]]+?)\s+((?:19|20)\d{2}),\s*p\.\s*(\d+)\]")

NO_EVIDENCE_ANSWER = (
    "I do not have enough information in the indexed documents to answer that question."
)

SYSTEM_INSTRUCTIONS = """You are a financial research assistant.

Rules:
1. Answer using ONLY the numbered context below.
2. If the context does not support an answer, reply exactly: INSUFFICIENT_EVIDENCE
   followed by one sentence explaining what is missing.
3. Cite every factual claim using the exact citation shown above the source you used,
   for example [Northstar Industrial 2024, p.18].
4. Do not invent citations. Only use citations that appear in the context.
5. Be concise: at most six sentences."""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(
        self,
        input_price_per_million: float | None,
        output_price_per_million: float | None,
    ) -> float | None:
        if input_price_per_million is None or output_price_per_million is None:
            return None
        return (
            self.input_tokens * input_price_per_million
            + self.output_tokens * output_price_per_million
        ) / 1_000_000


@dataclass
class CitationReport:
    cited: List[str] = field(default_factory=list)
    supported: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)
    unused_sources: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.unsupported

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)


@dataclass
class GenerationResult:
    answer: str
    usage: TokenUsage
    citations: CitationReport
    model: str
    called_llm: bool
    refused: bool


def format_context(retrieved: Sequence[RetrievalResult]) -> str:
    blocks = []
    for result in retrieved:
        blocks.append(
            f"SOURCE {result.rank} {result.chunk.citation}\n{result.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, retrieved: Sequence[RetrievalResult]) -> str:
    return f"""{SYSTEM_INSTRUCTIONS}

CONTEXT:
{format_context(retrieved)}

QUESTION:
{question}
"""


def extract_citations(answer: str) -> List[str]:
    """Normalised citation strings found in the answer text, in order, de-duplicated."""
    found: List[str] = []
    for company, year, page in CITATION_RE.findall(answer):
        citation = f"[{company.strip()} {year}, p.{page}]"
        if citation not in found:
            found.append(citation)
    return found


def validate_citations(answer: str, retrieved: Sequence[RetrievalResult]) -> CitationReport:
    """Check that every citation in the answer was actually supplied as context.

    This is a cheap, deterministic guard against the most visible RAG failure: a plausible
    citation pointing at a page the model never saw.
    """
    available = [result.chunk.citation for result in retrieved]
    cited = extract_citations(answer)

    return CitationReport(
        cited=cited,
        supported=[c for c in cited if c in available],
        unsupported=[c for c in cited if c not in available],
        unused_sources=[c for c in available if c not in cited],
    )


def _pricing_from_env() -> Tuple[float | None, float | None]:
    def _read(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    return _read("INPUT_PRICE_PER_1M_TOKENS"), _read("OUTPUT_PRICE_PER_1M_TOKENS")


def pricing() -> Tuple[float | None, float | None]:
    """Token prices in USD per million tokens, or (None, None) if not configured."""
    return _pricing_from_env()


def _usage_from_response(response) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def generate_answer(
    question: str,
    retrieved: Sequence[RetrievalResult],
    timeout: float = 30.0,
    max_retries: int = 2,
    client=None,
) -> GenerationResult:
    """Send the retrieved context to the LLM and validate what comes back.

    If retrieval returned nothing, no request is made: there is no evidence to ground an
    answer in, and calling the model would only cost tokens and invite invention.

    ``client`` exists so tests can inject a stub instead of reaching the network.
    """
    model = os.getenv("OPENAI_MODEL", "").strip()

    if not retrieved:
        return GenerationResult(
            answer=NO_EVIDENCE_ANSWER,
            usage=TokenUsage(),
            citations=CitationReport(),
            model=model or "none",
            called_llm=False,
            refused=True,
        )

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key or not model:
            raise RuntimeError(
                "OPENAI_API_KEY or OPENAI_MODEL is missing. "
                "Copy .env.example to .env and set both."
            )

        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    prompt = build_prompt(question, retrieved)

    response = _create_response(client, model, prompt)

    answer = (response.output_text or "").strip()

    return GenerationResult(
        answer=answer,
        usage=_usage_from_response(response),
        citations=validate_citations(answer, retrieved),
        model=model,
        called_llm=True,
        refused=answer.upper().startswith("INSUFFICIENT_EVIDENCE"),
    )


def _create_response(client, model: str, prompt: str):
    """Ask for temperature=0, but tolerate models that reject the parameter.

    Deterministic decoding is the right default for a grounded-answer system; several
    reasoning models reject the parameter outright, so falling back is more useful than
    failing.
    """
    try:
        return client.responses.create(model=model, input=prompt, temperature=0)
    except Exception as exc:  # noqa: BLE001 - narrow by message, not by class
        message = str(exc).lower()
        if "temperature" not in message:
            raise
        return client.responses.create(model=model, input=prompt)
