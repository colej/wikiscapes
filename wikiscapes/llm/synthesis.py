"""Multi-article synthesis: parse LLM responses for answers, sources, and gaps."""

from __future__ import annotations

import re

from wikiscapes.models import Article, QueryResult


def parse_synthesis_response(
    raw_response: str,
    query_coord: tuple[float, float] | None,
    cluster_hint: str,
    token_usage: dict[str, int],
) -> QueryResult:
    """Parse the synthesis LLM response into a QueryResult.

    Extracts:
    - answer: everything before the SOURCES: line
    - sources: article ids from SOURCES: line
    - gap_detected: True if GAP_DETECTED: line is present
    """
    lines = raw_response.strip().splitlines()

    sources: list[str] = []
    gap_detected = False
    answer_lines: list[str] = []
    gap_description = ""

    for line in lines:
        if line.startswith("SOURCES:"):
            raw_ids = line[len("SOURCES:"):].strip()
            sources = [s.strip() for s in raw_ids.split(",") if s.strip()]
        elif line.startswith("GAP_DETECTED:"):
            gap_detected = True
            gap_description = line[len("GAP_DETECTED:"):].strip()
        else:
            answer_lines.append(line)

    answer = "\n".join(answer_lines).strip()
    if gap_description:
        answer += f"\n\n> **Knowledge gap detected:** {gap_description}"

    return QueryResult(
        answer=answer,
        sources=sources,
        query_coord=query_coord,
        cluster_hint=cluster_hint,
        gap_detected=gap_detected,
        token_usage=token_usage,
    )


def build_token_budget(
    articles: list[Article],
    max_context_tokens: int,
    chars_per_token: int = 4,
) -> list[Article]:
    """Return a subset of articles that fit within max_context_tokens.

    Articles are assumed to be pre-ranked (best first). We include as many
    as fit within the token budget, always keeping at least the top 2.
    """
    budget_chars = max_context_tokens * chars_per_token
    selected: list[Article] = []
    used_chars = 0

    for i, article in enumerate(articles):
        body_chars = len(article.body)
        if i < 2 or used_chars + body_chars <= budget_chars:
            selected.append(article)
            used_chars += body_chars
        else:
            # Still include abbreviated version (title + key concepts ~200 chars)
            used_chars += 200
            selected.append(article)
            if used_chars >= budget_chars:
                break

    return selected


def extract_update_directive(response: str) -> tuple[str | None, str]:
    """Check if the ingest response starts with UPDATE:<existing_id>.

    Returns (existing_id, body) or (None, body).
    """
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    match = re.match(r"^UPDATE:(\S+)$", first_line)
    if match:
        existing_id = match.group(1)
        body = "\n".join(response.strip().splitlines()[1:]).strip()
        return existing_id, body
    return None, response.strip()
