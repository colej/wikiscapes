"""Manages wiki/index.md — the flat content catalog used for LLM context."""

from __future__ import annotations

from pathlib import Path

from wikiscapes.models import Article


_HEADER = """\
# Wiki Index

This catalog is auto-maintained by wikiscapes. One row per article.
Pass this file as context when ingesting or querying.

| id | title | cluster | kind | abstraction | summary |
|----|-------|---------|------|-------------|---------|
"""


def rebuild_index(articles: list[Article], wiki_dir: Path) -> None:
    """Rewrite wiki/index.md from the current article list."""
    index_path = wiki_dir / "index.md"
    rows = [index_entry_for(a) for a in sorted(articles, key=lambda a: a.frontmatter.id)]
    content = _HEADER + "\n".join(rows) + "\n"
    index_path.write_text(content, encoding="utf-8")


def read_index(wiki_dir: Path) -> str:
    """Return the full text of wiki/index.md, or an empty-index stub."""
    index_path = wiki_dir / "index.md"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return _HEADER


def index_entry_for(article: Article) -> str:
    """Format one markdown table row for the given article."""
    fm = article.frontmatter
    cluster = fm.topo.cluster_label if fm.topo else "unplaced"
    summary = _extract_summary(article.body)
    abstraction = f"{fm.abstraction_level:.2f}"
    # Escape pipe chars inside cell values
    cells = [
        _esc(fm.id),
        _esc(fm.title),
        _esc(cluster),
        _esc(fm.kind),
        abstraction,
        _esc(summary),
    ]
    return "| " + " | ".join(cells) + " |"


def _extract_summary(body: str, max_chars: int = 120) -> str:
    """Extract first non-heading, non-empty line as a one-line summary."""
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:max_chars] + ("…" if len(line) > max_chars else "")
    return ""


def _esc(text: str) -> str:
    return text.replace("|", "\\|")
