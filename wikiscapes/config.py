"""Configuration loading from wikiscapes.toml and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse-def]


@dataclass
class WikiConfig:
    wiki_dir: str = "wiki"
    raw_dir: str = "raw"
    state_dir: str = ".wikiscapes"
    index_filename: str = "index.md"


@dataclass
class LLMConfig:
    generation_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5"
    max_tokens_synthesis: int = 4096
    max_tokens_ingest: int = 2048
    max_context_tokens: int = 80_000


@dataclass
class EmbeddingsConfig:
    backend: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.05
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class ClusteringConfig:
    min_cluster_size: int = 5
    min_samples: int = 3


@dataclass
class TopologyConfig:
    neighborhood_radius: float = 0.15
    neighborhood_radius_expanded: float = 0.25
    k_fallback: int = 8
    k_local_connections: int = 5
    confluence_bandwidth: float = 0.12
    confluence_min_density: float = 0.3


@dataclass
class PlasticityConfig:
    importance_weight_max: int = 3
    evolve_article_change_threshold: float = 0.20
    evolve_query_log_threshold: int = 100
    access_log_flush_interval: int = 10


@dataclass
class Config:
    wiki: WikiConfig = field(default_factory=WikiConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    # Root of the wiki project (directory containing wikiscapes.toml)
    project_root: Path = field(default_factory=Path.cwd)

    @property
    def anthropic_api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
        return key

    @property
    def openai_api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        return key

    def wiki_dir_path(self) -> Path:
        return self.project_root / self.wiki.wiki_dir

    def raw_dir_path(self) -> Path:
        return self.project_root / self.wiki.raw_dir

    def state_dir_path(self) -> Path:
        return self.project_root / self.wiki.state_dir

    def index_path(self) -> Path:
        return self.wiki_dir_path() / self.wiki.index_filename


def load_config(path: Path | None = None) -> Config:
    """Load config from wikiscapes.toml, falling back to defaults.

    Environment variable overrides follow the pattern:
        WIKISCAPES_LLM__GENERATION_MODEL  (double underscore = nesting)
    """
    if path is None:
        path = Path.cwd() / "wikiscapes.toml"

    project_root = path.parent

    raw: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            raw = tomllib.load(f)

    config = Config(project_root=project_root)

    if "wiki" in raw:
        w = raw["wiki"]
        config.wiki = WikiConfig(
            wiki_dir=w.get("wiki_dir", config.wiki.wiki_dir),
            raw_dir=w.get("raw_dir", config.wiki.raw_dir),
            state_dir=w.get("state_dir", config.wiki.state_dir),
            index_filename=w.get("index_filename", config.wiki.index_filename),
        )

    if "llm" in raw:
        l = raw["llm"]
        config.llm = LLMConfig(
            generation_model=l.get("generation_model", config.llm.generation_model),
            fast_model=l.get("fast_model", config.llm.fast_model),
            max_tokens_synthesis=l.get("max_tokens_synthesis", config.llm.max_tokens_synthesis),
            max_tokens_ingest=l.get("max_tokens_ingest", config.llm.max_tokens_ingest),
            max_context_tokens=l.get("max_context_tokens", config.llm.max_context_tokens),
        )

    if "embeddings" in raw:
        e = raw["embeddings"]
        config.embeddings = EmbeddingsConfig(
            backend=e.get("backend", config.embeddings.backend),
            model=e.get("model", config.embeddings.model),
            batch_size=e.get("batch_size", config.embeddings.batch_size),
        )

    if "umap" in raw:
        u = raw["umap"]
        config.umap = UMAPConfig(
            n_neighbors=u.get("n_neighbors", config.umap.n_neighbors),
            min_dist=u.get("min_dist", config.umap.min_dist),
            metric=u.get("metric", config.umap.metric),
            random_state=u.get("random_state", config.umap.random_state),
        )

    if "clustering" in raw:
        c = raw["clustering"]
        config.clustering = ClusteringConfig(
            min_cluster_size=c.get("min_cluster_size", config.clustering.min_cluster_size),
            min_samples=c.get("min_samples", config.clustering.min_samples),
        )

    if "topology" in raw:
        t = raw["topology"]
        config.topology = TopologyConfig(
            neighborhood_radius=t.get("neighborhood_radius", config.topology.neighborhood_radius),
            neighborhood_radius_expanded=t.get(
                "neighborhood_radius_expanded", config.topology.neighborhood_radius_expanded
            ),
            k_fallback=t.get("k_fallback", config.topology.k_fallback),
            k_local_connections=t.get(
                "k_local_connections", config.topology.k_local_connections
            ),
            confluence_bandwidth=t.get(
                "confluence_bandwidth", config.topology.confluence_bandwidth
            ),
            confluence_min_density=t.get(
                "confluence_min_density", config.topology.confluence_min_density
            ),
        )

    if "plasticity" in raw:
        p = raw["plasticity"]
        config.plasticity = PlasticityConfig(
            importance_weight_max=p.get(
                "importance_weight_max", config.plasticity.importance_weight_max
            ),
            evolve_article_change_threshold=p.get(
                "evolve_article_change_threshold",
                config.plasticity.evolve_article_change_threshold,
            ),
            evolve_query_log_threshold=p.get(
                "evolve_query_log_threshold", config.plasticity.evolve_query_log_threshold
            ),
            access_log_flush_interval=p.get(
                "access_log_flush_interval", config.plasticity.access_log_flush_interval
            ),
        )

    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: Config) -> None:
    """Apply WIKISCAPES_SECTION__KEY environment variable overrides."""
    prefix = "WIKISCAPES_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        if "__" not in remainder:
            continue
        section, attr = remainder.split("__", 1)
        section = section.lower()
        attr = attr.lower()
        section_obj = getattr(config, section, None)
        if section_obj is not None and hasattr(section_obj, attr):
            current = getattr(section_obj, attr)
            try:
                if isinstance(current, bool):
                    setattr(section_obj, attr, value.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(section_obj, attr, int(value))
                elif isinstance(current, float):
                    setattr(section_obj, attr, float(value))
                else:
                    setattr(section_obj, attr, value)
            except (ValueError, TypeError):
                pass


DEFAULT_TOML = """\
[wiki]
wiki_dir = "wiki"
raw_dir = "raw"
state_dir = ".wikiscapes"

[llm]
generation_model = "claude-sonnet-4-6"
fast_model = "claude-haiku-4-5"
max_context_tokens = 80000

[embeddings]
backend = "sentence-transformers"   # or "openai"
model = "all-MiniLM-L6-v2"
batch_size = 32

[umap]
n_neighbors = 15
min_dist = 0.05
metric = "cosine"
random_state = 42

[clustering]
min_cluster_size = 5
min_samples = 3

[topology]
neighborhood_radius = 0.15
neighborhood_radius_expanded = 0.25
k_fallback = 8
k_local_connections = 5
confluence_bandwidth = 0.12
confluence_min_density = 0.3

[plasticity]
importance_weight_max = 3
evolve_article_change_threshold = 0.20
evolve_query_log_threshold = 100
access_log_flush_interval = 10
"""
