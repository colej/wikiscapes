"""All LLM prompt templates for wikiscapes.

Keeping prompts here (not scattered across pipeline modules) enables:
- Versioning and A/B testing independently of pipeline logic
- Easier testing (prompts are pure functions)
- Clear separation of concerns

Model-tier guidance:
- INGEST_SYSTEM_PROMPT / ingest_prompt → Sonnet (full, cache_system=True)
- SYNTHESIS_SYSTEM_PROMPT / query_prompt → Sonnet (full, cache_system=True)
- BRIDGE_ARTICLE_SYSTEM_PROMPT / bridge_article_prompt → Sonnet (full)
- All Haiku prompts: cluster labels, kind classification, abstraction scoring,
  lint neighbor validation — these are inline one-liners in their modules.
"""

from __future__ import annotations

from wikiscapes.models import Article

# ---------------------------------------------------------------------------
# System prompts (candidates for prompt caching — sent repeatedly)
# ---------------------------------------------------------------------------

INGEST_SYSTEM_PROMPT = """\
You are a wiki article author for a personal knowledge base.
Given a source text and a catalog of existing articles, write a structured \
markdown wiki article.

Rules:
- Title: clear, encyclopedic, specific (not generic)
- Body: 300–800 words unless the source is very long or technical
- Use ## subheadings to organize content
- Do not invent facts beyond what the source provides
- End with a ## Key Concepts section containing 5–10 bullet-point terms
- Check the provided index: if this topic already exists, expand that article \
  instead of creating a duplicate; respond with UPDATE:<existing_id> on the \
  first line followed by the updated article body
- Write in third-person encyclopedic style
- Do not include the YAML frontmatter block — only the markdown body
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a research assistant synthesizing answers from a curated knowledge base.

Rules:
- Answer solely from the provided articles; do not introduce outside knowledge
- If the articles do not fully cover the question, say so explicitly and note \
  what additional information would be needed (this signals a knowledge gap)
- Cite the article IDs you draw from, e.g. [quantum-entanglement]
- Structure longer answers with ## headings
- If you detect contradictions between articles, flag them
- End your response with: SOURCES: <comma-separated article ids used>
- If the question cannot be answered from the provided articles, end with: \
  GAP_DETECTED: <brief description of missing knowledge>
"""

BRIDGE_ARTICLE_SYSTEM_PROMPT = """\
You are a wiki article author specializing in cross-domain synthesis.
You will be given articles from two distinct topic clusters that share a \
geographic region on a topographic knowledge map. Write a bridge article that \
synthesizes the connections between these domains.

Rules:
- Title: frame the connection explicitly, e.g. "Quantum Computing and \
  Cryptography: Intersection and Implications"
- Body: 400–700 words
- Use ## subheadings
- Focus on genuine conceptual bridges, not superficial similarities
- End with ## Bridge Concepts — 5–8 bullet points naming the specific \
  ideas that connect the two domains
- Do not include YAML frontmatter — only the markdown body
"""

# ---------------------------------------------------------------------------
# Prompt factories (user-turn messages)
# ---------------------------------------------------------------------------

def ingest_prompt(source_text: str, existing_index: str, article_id: str) -> str:
    """Build the user-turn message for ingesting a new source."""
    return (
        f"Article ID to create: {article_id}\n\n"
        f"## Existing Wiki Index\n\n{existing_index}\n\n"
        f"## Source Text\n\n{source_text}"
    )


def query_prompt(
    question: str,
    retrieved_articles: list[Article],
    map_context: str,
) -> str:
    """Build the user-turn message for synthesizing an answer.

    Context window strategy (token optimization):
    - Full body for top-2 articles
    - Title + ## Key Concepts only for articles 3+
    """
    parts: list[str] = [
        f"## Map Context\n{map_context}\n",
        f"## Question\n{question}\n",
        "## Retrieved Articles\n",
    ]

    for i, article in enumerate(retrieved_articles):
        fm = article.frontmatter
        if i < 2:
            parts.append(f"### [{fm.id}] {fm.title}\n\n{article.body}\n")
        else:
            # Abbreviated: title + Key Concepts section only
            key_concepts = _extract_key_concepts(article.body)
            parts.append(
                f"### [{fm.id}] {fm.title} (summary)\n\n{key_concepts}\n"
            )

    return "\n".join(parts)


def bridge_article_prompt(
    cluster_a_label: str,
    cluster_b_label: str,
    bridge_articles: list[Article],
    bridge_article_id: str,
) -> str:
    """Build the user-turn message for generating a bridge article."""
    # Truncate each context article to 1500 tokens worth of chars (~6000 chars)
    _MAX_CHARS = 6000
    article_blocks = []
    for a in bridge_articles:
        body_excerpt = a.body[:_MAX_CHARS]
        article_blocks.append(
            f"### [{a.frontmatter.id}] {a.frontmatter.title}\n\n{body_excerpt}"
        )

    return (
        f"Bridge Article ID to create: {bridge_article_id}\n\n"
        f"Cluster A: {cluster_a_label}\n"
        f"Cluster B: {cluster_b_label}\n\n"
        f"## Context Articles\n\n" + "\n\n".join(article_blocks)
    )


def stub_expansion_prompt(stub: Article, neighbor_articles: list[Article]) -> str:
    """Build the user-turn message for expanding a stub into a full article."""
    _MAX_CHARS = 4000
    neighbor_blocks = [
        f"### [{a.frontmatter.id}] {a.frontmatter.title}\n\n{a.body[:_MAX_CHARS]}"
        for a in neighbor_articles[:4]
    ]
    return (
        f"Expand this stub wiki article into a full article (300–800 words).\n\n"
        f"## Stub: [{stub.frontmatter.id}] {stub.frontmatter.title}\n\n"
        f"{stub.body}\n\n"
        f"## Topographic Neighbors (for context)\n\n"
        + "\n\n".join(neighbor_blocks)
    )


def cluster_label_prompt(titles: list[str]) -> str:
    """One-liner prompt for Haiku cluster labeling."""
    titles_str = "\n".join(f"- {t}" for t in titles[:20])
    return (
        f"These wiki articles belong to the same topic cluster:\n{titles_str}\n\n"
        "Give this cluster a clear, concise label of 2–3 words that captures "
        "the shared topic. Respond with ONLY the label, nothing else."
    )


def kind_classification_prompt(title: str, first_paragraph: str) -> str:
    """Haiku prompt: classify article kind."""
    return (
        f"Classify this wiki article as one of: factual, synthesized, bridge, stub\n\n"
        f"Title: {title}\n\n{first_paragraph[:500]}\n\n"
        "Respond with ONLY one word: factual, synthesized, bridge, or stub."
    )


def abstraction_level_prompt(title: str, first_paragraph: str) -> str:
    """Haiku prompt: score abstraction level 0.0–1.0."""
    return (
        f"Rate how abstract or conceptual this wiki article is on a scale from "
        f"0.0 (very concrete, factual, specific) to 1.0 (very abstract, theoretical, meta).\n\n"
        f"Title: {title}\n\n{first_paragraph[:500]}\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_key_concepts(body: str) -> str:
    """Extract the ## Key Concepts section from an article body, or first 300 chars."""
    lines = body.splitlines()
    in_key_concepts = False
    section_lines: list[str] = []

    for line in lines:
        if line.strip().lower().startswith("## key concepts"):
            in_key_concepts = True
            section_lines.append(line)
            continue
        if in_key_concepts:
            if line.startswith("## ") and not line.strip().lower().startswith("## key"):
                break
            section_lines.append(line)

    if section_lines:
        return "\n".join(section_lines)
    return body[:300] + "…"
