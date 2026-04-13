"""Pydantic data models shared across all wikiscapes modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TopoPosition(BaseModel):
    x: float
    y: float
    cluster_id: str
    cluster_label: str
    embedding_version: str


class ArticleConnections(BaseModel):
    local: list[str] = Field(default_factory=list)
    distant: list[str] = Field(default_factory=list)
    # Only populated on bridge articles
    confluence_of: list[str] = Field(default_factory=list)


class ArticleAccess(BaseModel):
    count: int = 0
    last_accessed: datetime | None = None
    # Last 5 query strings that retrieved this article
    query_contexts: list[str] = Field(default_factory=list)


class ArticleFrontmatter(BaseModel):
    id: str
    title: str
    created: datetime
    updated: datetime
    sources: list[str] = Field(default_factory=list)
    # None until first layout run
    topo: TopoPosition | None = None
    connections: ArticleConnections = Field(default_factory=ArticleConnections)
    access: ArticleAccess = Field(default_factory=ArticleAccess)
    kind: Literal["factual", "synthesized", "bridge", "stub"] = "stub"
    # 0.0 = concrete/factual, 1.0 = abstract/meta
    abstraction_level: float = 0.5
    # Computed: sigmoid((access_count / max_access) * recency_weight)
    importance: float = 0.0


class Article(BaseModel):
    frontmatter: ArticleFrontmatter
    # Raw markdown body (frontmatter stripped)
    body: str
    path: Path


class ConfluenceZone(BaseModel):
    id: str
    centroid_x: float
    centroid_y: float
    # cluster_ids that overlap at this zone
    clusters: list[str]
    radius: float
    density: float
    # Set after a bridge article is generated
    bridge_article_id: str | None = None


class MapState(BaseModel):
    # article_id → position
    articles: dict[str, TopoPosition] = Field(default_factory=dict)
    # cluster_id → [article_ids]
    clusters: dict[str, list[str]] = Field(default_factory=dict)
    confluence_zones: list[ConfluenceZone] = Field(default_factory=list)
    layout_version: int = 0
    article_count_at_layout: int = 0
    embedding_model: str = ""
    umap_params: dict[str, Any] = Field(default_factory=dict)
    created: datetime = Field(default_factory=datetime.utcnow)
    last_evolved: datetime = Field(default_factory=datetime.utcnow)


class AccessLogEntry(BaseModel):
    timestamp: datetime
    query: str
    articles_hit: list[str]
    query_coord: tuple[float, float] | None = None


class QueryResult(BaseModel):
    answer: str
    sources: list[str]
    query_coord: tuple[float, float] | None = None
    cluster_hint: str = ""
    gap_detected: bool = False
    token_usage: dict[str, int] = Field(default_factory=dict)


class LintIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str
    article_id: str | None = None
    message: str


class LintReport(BaseModel):
    total_articles: int
    issues: list[LintIssue] = Field(default_factory=list)
    topology_health: float = 1.0
    map_coverage: float = 0.0
    cluster_balance: dict[str, int] = Field(default_factory=dict)
    stubs: list[str] = Field(default_factory=list)
    orphans: list[str] = Field(default_factory=list)
    # (x, y) grid centers of empty map regions
    empty_regions: list[tuple[float, float]] = Field(default_factory=list)


class EvolveResult(BaseModel):
    articles_re_embedded: int
    clusters_changed: int
    bridges_generated: int
    layout_version: int
    duration_seconds: float
