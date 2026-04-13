# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**wikiscapes** is a Python CLI tool that layers Brainscapes-inspired topographic organization on top of Karpathy's LLM Wiki architecture. Instead of a flat index, it maintains a living 2D UMAP map of the knowledge base where semantic proximity = spatial proximity. Articles are routed, discovered, and grown using principles from cortical topographic maps.

## Commands

```bash
# Install (development)
uv sync

# Run CLI
uv run wikiscapes --help
uv run wikiscapes init             # Initialize a new wiki project
uv run wikiscapes ingest <file>    # Ingest a raw source file
uv run wikiscapes query "..."      # Query the wiki
uv run wikiscapes evolve           # Re-layout the topographic map
uv run wikiscapes lint [--deep]    # Health checks
uv run wikiscapes map [--format ascii|png|html]
uv run wikiscapes status

# Tests
uv run pytest
```

## Architecture

### Data Flow
```
raw/           →  ingest  →  wiki/{id}.md  →  evolve  →  .wikiscapes/
(sources)         (Sonnet)   (articles)        (UMAP)     (map state)
                                                  ↓
                                            query → synthesis (Sonnet)
```

### Key Directories
- `wiki/` — LLM-maintained markdown articles with YAML frontmatter
- `raw/` — Immutable source materials
- `.wikiscapes/` — Internal state: `map_state.json`, `embeddings.npy`, `kdtree.pkl`, `access_log.jsonl`

### Package Structure (`wikiscapes/`)
- `models.py` — All Pydantic models (`Article`, `MapState`, `QueryResult`, etc.). Every module imports from here.
- `config.py` — Loads `wikiscapes.toml` + env var overrides. Single `Config` dataclass passed everywhere.
- `store/` — Disk I/O: `article_store.py` (markdown+frontmatter), `index.py` (index.md), `map_state.py` (JSON/numpy/pickle)
- `topo/` — Topographic engine: `embed.py` (sentence-transformers or openai), `layout.py` (UMAP + HDBSCAN + Homunculus), `spatial.py` (KDTree queries), `confluence.py` (KDE-based zone detection)
- `llm/` — LLM layer: `client.py` (Anthropic SDK wrapper), `prompts.py` (all prompt templates), `synthesis.py` (response parsing)
- `core/` — Pipelines: `ingest.py`, `query.py`, `evolve.py`, `lint.py`
- `viz/` — `plotly_map.py` (interactive HTML), `static_map.py` (PNG + ASCII)
- `cli.py` — Typer CLI entry point

### Model Routing
- **Sonnet** (`tier="full"`): ingest article generation, query synthesis, bridge article generation
- **Haiku** (`tier="fast"`): cluster labels, lint neighbor validation, kind/abstraction classification
- Prompt caching (`cache_system=True`) applied to `INGEST_SYSTEM_PROMPT` and `SYNTHESIS_SYSTEM_PROMPT` (high-repetition Sonnet calls)

### Article Frontmatter Schema
Every `wiki/*.md` file has YAML frontmatter managed by the engine:
```yaml
id: "article-slug"
title: "Article Title"
sources: ["raw/source.txt"]
kind: "factual"   # factual | synthesized | bridge | stub
abstraction_level: 0.15   # 0=concrete, 1=abstract
topo:
  x: 0.42; y: 0.71
  cluster_id: "quantum-physics"
  cluster_label: "Quantum Physics"
  embedding_version: "1"
connections:
  local: ["neighbor-id-1", "neighbor-id-2"]
  distant: []
  confluence_of: []   # only on bridge articles
access:
  count: 47
  last_accessed: "..."
  query_contexts: [...]
importance: 0.82
```

### Query Routing Algorithm (6 phases)
1. Embed query → full-dim vector
2. Project → 2D map coordinate (weighted centroid of k-nearest neighbors in embedding space)
3. Radius search on KDTree → candidate article IDs
4. Rerank by full-dim cosine similarity (corrects UMAP distortion)
5. Token-budgeted synthesis: full body for top-2, title+## Key Concepts for articles 3+
6. Log to `access_log.jsonl`, batch-flush access counts every N queries

### Brainscapes Principles Applied
- **Proximity = Relatedness**: UMAP clusters semantically similar articles spatially
- **Homunculus**: High-importance articles duplicated in embedding matrix before UMAP → expand on map
- **Confluence zones**: KDE detects where cluster boundaries overlap → triggers bridge articles
- **Plasticity**: Access log drives periodic re-layout; `should_suggest_evolve()` checks thresholds

## Environment Variables
- `ANTHROPIC_API_KEY` — required for all LLM operations
- `OPENAI_API_KEY` — required only if `embeddings.backend = "openai"` in wikiscapes.toml
- `WIKISCAPES_LLM__GENERATION_MODEL` — override any config value (double underscore = nesting)
