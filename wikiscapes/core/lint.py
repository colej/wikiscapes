"""Lint pipeline: structural, topographic, and optional semantic health checks.

Structural checks (free, always run):
- YAML parse errors, missing sources, broken connection references
- Duplicate IDs, stale embedding_version, index completeness

Topographic checks (free, uses MapState):
- Orphans (noise cluster -1), map gap detection, cluster balance
- Stub ratio, confluence zones without bridge articles

Semantic checks (Haiku, --deep flag only):
- Neighbor-pair topographic validation (5 random pairs)
- Stale articles (source files newer than frontmatter.updated)
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from wikiscapes.models import LintIssue, LintReport
from wikiscapes.store.article_store import load_all_articles
from wikiscapes.store.index import read_index
from wikiscapes.store.map_state import load_map_state

if TYPE_CHECKING:
    from wikiscapes.config import Config
    from wikiscapes.llm.client import WikiClient


def lint(
    config: Config,
    client: WikiClient | None = None,
    fix: bool = False,
    deep: bool = False,
) -> LintReport:
    """Run all applicable health checks.

    Args:
        config: Project config.
        client: WikiClient — required only if deep=True.
        fix: If True, attempt auto-repair for fixable issues.
        deep: If True, run Haiku-based semantic checks.

    Returns LintReport.
    """
    wiki_dir = config.wiki_dir_path()
    state_dir = config.state_dir_path()
    raw_dir = config.raw_dir_path()

    articles = load_all_articles(wiki_dir)
    map_state = load_map_state(state_dir)
    issues: list[LintIssue] = []

    # --- Structural checks ---
    article_ids = {a.frontmatter.id for a in articles}

    for article in articles:
        fm = article.frontmatter

        # Missing source files
        for src in fm.sources:
            src_path = Path(src)
            if not src_path.exists() and not (raw_dir / src_path.name).exists():
                issues.append(LintIssue(
                    severity="warning",
                    code="MISSING_SOURCE",
                    article_id=fm.id,
                    message=f"Source not found: {src}",
                ))

        # Broken connections.local references
        for ref in fm.connections.local:
            if ref not in article_ids:
                issues.append(LintIssue(
                    severity="error",
                    code="BROKEN_LOCAL_CONNECTION",
                    article_id=fm.id,
                    message=f"connections.local references missing article: {ref}",
                ))
                if fix:
                    _fix_broken_connection(article, ref, wiki_dir)

        # Stale embedding version
        if fm.topo and fm.topo.embedding_version != "1":
            issues.append(LintIssue(
                severity="info",
                code="STALE_EMBEDDING",
                article_id=fm.id,
                message=f"Embedding version {fm.topo.embedding_version!r} != current '1'",
            ))

    # Duplicate IDs
    seen_ids: dict[str, str] = {}
    for article in articles:
        aid = article.frontmatter.id
        if aid in seen_ids:
            issues.append(LintIssue(
                severity="error",
                code="DUPLICATE_ID",
                article_id=aid,
                message=f"Duplicate article ID found at {article.path} and {seen_ids[aid]}",
            ))
        else:
            seen_ids[aid] = str(article.path)

    # Index completeness
    index_text = read_index(wiki_dir)
    for article in articles:
        if article.frontmatter.id not in index_text:
            issues.append(LintIssue(
                severity="warning",
                code="MISSING_FROM_INDEX",
                article_id=article.frontmatter.id,
                message="Article not found in index.md",
            ))
            if fix:
                from wikiscapes.store.index import rebuild_index
                rebuild_index(articles, wiki_dir)
                break  # rebuilding index fixes all at once

    # --- Topographic checks ---
    stubs = [a.frontmatter.id for a in articles if a.frontmatter.kind == "stub"]
    orphans: list[str] = []
    empty_regions: list[tuple[float, float]] = []
    cluster_balance: dict[str, int] = {}
    topology_health = 1.0
    map_coverage = 0.0
    confluence_without_bridge: list[str] = []

    if map_state is not None:
        # Orphans: articles in noise cluster
        for article in articles:
            if article.frontmatter.topo and article.frontmatter.topo.cluster_id == "unplaced":
                orphans.append(article.frontmatter.id)
                issues.append(LintIssue(
                    severity="warning",
                    code="ORPHAN_ARTICLE",
                    article_id=article.frontmatter.id,
                    message="Article is in the noise cluster (unplaced).",
                ))

        # Cluster balance
        for cluster_id, aids in map_state.clusters.items():
            cluster_balance[cluster_id] = len(aids)

        total = sum(cluster_balance.values())
        if total > 0:
            for cid, count in cluster_balance.items():
                if count / total > 0.40 and len(cluster_balance) > 1:
                    issues.append(LintIssue(
                        severity="warning",
                        code="CLUSTER_IMBALANCE",
                        message=f"Cluster '{cid}' holds {count/total:.0%} of articles.",
                    ))

        # Stub ratio
        if articles:
            stub_ratio = len(stubs) / len(articles)
            if stub_ratio > 0.20:
                issues.append(LintIssue(
                    severity="warning",
                    code="HIGH_STUB_RATIO",
                    message=f"{stub_ratio:.0%} of articles are stubs (threshold: 20%).",
                ))

        # Gap detection: 10×10 grid scan
        import numpy as np
        grid = 10
        occupied = set()
        for pos in map_state.articles.values():
            cell_x = min(int(pos.x * grid), grid - 1)
            cell_y = min(int(pos.y * grid), grid - 1)
            occupied.add((cell_x, cell_y))

        total_cells = grid * grid
        empty_cells = [(cx, cy) for cx in range(grid) for cy in range(grid) if (cx, cy) not in occupied]
        empty_regions = [
            (cx / grid + 0.05, cy / grid + 0.05) for cx, cy in empty_cells
        ]
        map_coverage = len(occupied) / total_cells

        # Confluence zones without bridges
        for zone in map_state.confluence_zones:
            if zone.bridge_article_id is None:
                confluence_without_bridge.append(zone.id)
                issues.append(LintIssue(
                    severity="info",
                    code="CONFLUENCE_NO_BRIDGE",
                    message=f"Confluence zone {zone.id} has no bridge article. Run evolve to generate.",
                ))

        # Topology health score
        n_issues = sum(1 for i in issues if i.severity == "error")
        n_warnings = sum(1 for i in issues if i.severity == "warning")
        topology_health = max(0.0, 1.0 - 0.1 * n_issues - 0.05 * n_warnings)

    # --- Semantic checks (Haiku, --deep) ---
    if deep and client is not None and len(articles) >= 2:
        sample_pairs = _sample_neighbor_pairs(articles, n=5)
        for a1, a2 in sample_pairs:
            verdict = _validate_neighbor_pair(a1, a2, client)
            if not verdict:
                issues.append(LintIssue(
                    severity="warning",
                    code="DUBIOUS_NEIGHBOR",
                    article_id=a1.frontmatter.id,
                    message=f"Topographic neighbor relationship with '{a2.frontmatter.id}' seems incorrect.",
                ))

        # Stale articles (sources newer than updated)
        for article in articles:
            for src in article.frontmatter.sources:
                src_path = Path(src)
                if not src_path.exists():
                    src_path = config.raw_dir_path() / src_path.name
                if src_path.exists():
                    src_mtime = datetime.fromtimestamp(
                        src_path.stat().st_mtime, tz=timezone.utc
                    )
                    if src_mtime > article.frontmatter.updated:
                        issues.append(LintIssue(
                            severity="info",
                            code="STALE_ARTICLE",
                            article_id=article.frontmatter.id,
                            message=f"Source {src_path.name} is newer than article (may need re-ingest).",
                        ))

    return LintReport(
        total_articles=len(articles),
        issues=issues,
        topology_health=topology_health,
        map_coverage=map_coverage,
        cluster_balance=cluster_balance,
        stubs=stubs,
        orphans=orphans,
        empty_regions=empty_regions,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_broken_connection(article: object, ref: str, wiki_dir: Path) -> None:
    """Remove a broken reference from connections.local."""
    from wikiscapes.models import Article
    from wikiscapes.store.article_store import save_article

    a: Article = article  # type: ignore
    updated_local = [r for r in a.frontmatter.connections.local if r != ref]
    conn = a.frontmatter.connections.model_copy(update={"local": updated_local})
    fm = a.frontmatter.model_copy(update={"connections": conn})
    save_article(Article(frontmatter=fm, body=a.body, path=a.path))


def _sample_neighbor_pairs(
    articles: list[object], n: int = 5
) -> list[tuple[object, object]]:
    """Return n random (article, neighbor) pairs for semantic validation."""
    from wikiscapes.models import Article

    arts: list[Article] = articles  # type: ignore
    pairs = []
    candidates = [a for a in arts if a.frontmatter.connections.local]
    random.shuffle(candidates)
    for a in candidates[:n]:
        neighbor_id = a.frontmatter.connections.local[0]
        neighbor = next((x for x in arts if x.frontmatter.id == neighbor_id), None)
        if neighbor:
            pairs.append((a, neighbor))
    return pairs


def _validate_neighbor_pair(a1: object, a2: object, client: object) -> bool:
    """Ask Haiku if two articles are correctly linked as topographic neighbors."""
    from wikiscapes.models import Article
    from wikiscapes.llm.client import WikiClient

    art1: Article = a1  # type: ignore
    art2: Article = a2  # type: ignore
    cli: WikiClient = client  # type: ignore

    prompt = (
        f"Are these two wiki articles topically related enough to be neighbors "
        f"in a knowledge map? Answer YES or NO only.\n\n"
        f"Article 1: {art1.frontmatter.title}\n"
        f"Article 2: {art2.frontmatter.title}"
    )
    response = cli.generate(prompt, tier="fast", max_tokens=5).strip().upper()
    return response.startswith("YES")
