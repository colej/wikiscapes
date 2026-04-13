"""Disk I/O for wiki articles — reading and writing markdown with YAML frontmatter."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from wikiscapes.models import Article, ArticleFrontmatter


def load_article(path: Path) -> Article:
    """Parse a wiki markdown file into an Article model."""
    post = frontmatter.load(str(path))
    fm_data = dict(post.metadata)
    fm = _parse_frontmatter(fm_data, path)
    return Article(frontmatter=fm, body=post.content, path=path)


def save_article(article: Article, path: Path | None = None) -> None:
    """Write an article back to disk, serializing frontmatter as YAML."""
    target = path or article.path
    target.parent.mkdir(parents=True, exist_ok=True)
    fm_dict = _frontmatter_to_dict(article.frontmatter)
    post = frontmatter.Post(article.body, **fm_dict)
    target.write_text(frontmatter.dumps(post), encoding="utf-8")


def load_all_articles(wiki_dir: Path) -> list[Article]:
    """Load every .md file in wiki_dir (excluding index.md)."""
    articles = []
    for md_path in sorted(wiki_dir.glob("*.md")):
        if md_path.name == "index.md":
            continue
        try:
            articles.append(load_article(md_path))
        except Exception as exc:
            # Surface parse errors rather than silently skipping
            raise ValueError(f"Failed to parse {md_path}: {exc}") from exc
    return articles


def find_article_by_id(article_id: str, wiki_dir: Path) -> Article | None:
    """Return the article with the given id slug, or None if not found."""
    path = wiki_dir / f"{article_id}.md"
    if path.exists():
        return load_article(path)
    return None


def update_frontmatter(article: Article, updates: dict) -> Article:
    """Return a new Article with frontmatter fields updated from a dict."""
    fm_dict = _frontmatter_to_dict(article.frontmatter)
    fm_dict.update(updates)
    fm_dict["updated"] = datetime.now(timezone.utc)
    new_fm = _parse_frontmatter(fm_dict, article.path)
    return Article(frontmatter=new_fm, body=article.body, path=article.path)


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "article"


def unique_article_id(base_id: str, wiki_dir: Path) -> str:
    """Return base_id if unused, otherwise append -v2, -v3, etc."""
    if not (wiki_dir / f"{base_id}.md").exists():
        return base_id
    counter = 2
    while (wiki_dir / f"{base_id}-v{counter}.md").exists():
        counter += 1
    return f"{base_id}-v{counter}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(data: dict, path: Path) -> ArticleFrontmatter:
    """Coerce raw YAML dict to ArticleFrontmatter, applying defaults."""
    now = datetime.now(timezone.utc)

    # Ensure datetime objects are timezone-aware
    def _ensure_tz(dt: object) -> datetime:
        if isinstance(dt, datetime):
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return now

    data.setdefault("id", path.stem)
    data.setdefault("title", path.stem.replace("-", " ").title())
    data["created"] = _ensure_tz(data.get("created", now))
    data["updated"] = _ensure_tz(data.get("updated", now))
    return ArticleFrontmatter.model_validate(data)


def _frontmatter_to_dict(fm: ArticleFrontmatter) -> dict:
    """Serialize ArticleFrontmatter to a plain dict for python-frontmatter."""
    d = fm.model_dump(exclude_none=False)
    # Convert nested models to plain dicts; pydantic already does this via model_dump
    # Convert datetime to ISO strings (python-frontmatter handles datetime natively
    # but explicit strings ensure cross-platform consistency)
    for key in ("created", "updated"):
        if isinstance(d.get(key), datetime):
            d[key] = d[key].isoformat()
    if d.get("access") and isinstance(d["access"].get("last_accessed"), datetime):
        d["access"]["last_accessed"] = d["access"]["last_accessed"].isoformat()
    return d
