# wikiscapes

**A Brainscapes-inspired topographic LLM knowledge base wiki.**

Wikiscapes extends [Andrej Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) concept with a living 2D topographic map of your knowledge base — inspired by the cortical maps described in Rebecca Schwarzschlose's *Brainscapes*. Instead of scanning a flat index to route queries, wikiscapes builds a spatial map where **semantic proximity equals spatial proximity**, enabling faster routing, organic discovery, and visual navigation of your growing knowledge base.

---

## The Core Ideas

### Karpathy's LLM Wiki

An LLM wiki treats your personal knowledge base like a compiled artifact:

- **`raw/`** — immutable source materials you feed in (papers, articles, notes, PDFs)
- **`wiki/`** — the LLM-compiled output: a collection of interconnected markdown articles
- **`index.md`** — a flat catalog of all articles, sized to fit in one context window

The LLM acts as a compiler. Raw sources go in, structured knowledge comes out. Unlike RAG, the knowledge is **stateful** — it compounds and cross-references over time rather than starting fresh on every query.

### Brainscapes Topography

The brain organizes sensory information in **topographic maps**: nearby neurons process related inputs. Adjacent points in the visual field are represented by adjacent neurons in visual cortex (retinotopy). This locality principle is not accidental — it minimizes wiring cost, enables efficient routing, and causes related concepts to cluster naturally through activity-dependent self-organization.

Wikiscapes applies the same principles to a knowledge base:

| Brain principle | Wikiscapes implementation |
|-----------------|--------------------------|
| Proximity = relatedness | UMAP places semantically similar articles spatially close |
| Cortical homunculus (important areas expand) | High-traffic articles weighted in UMAP → expand on map |
| Self-organized clustering | HDBSCAN finds clusters from layout, not imposed taxonomy |
| Confluence zones (cross-modal regions) | KDE detects where clusters overlap → bridge articles generated |
| Activity-dependent plasticity | Query access log drives periodic re-layout |
| Local connectivity | Articles link primarily to spatial neighbors |

### What this improves over a flat wiki

| | Flat LLM Wiki | Wikiscapes |
|---|---|---|
| Query routing | Scan entire index.md | Navigate to nearest map region |
| Discovery | Explicit search only | Nearby articles surface automatically |
| Knowledge gaps | Not visible | Empty map regions = missing domains |
| Navigation | Text search | Visual 2D topographic map |
| Cross-domain synthesis | Manual | Bridge articles auto-generated at confluence zones |
| Resource allocation | Uniform | High-traffic areas expand (Homunculus) |

### What it doesn't improve

- **Small wikis (<50 articles)**: topographic overhead isn't worth it at this scale
- **Cold start**: needs sufficient content to form meaningful topology
- **Exact-match / keyword queries**: flat index lookup is still faster
- **Cost**: adds embedding calls and periodic UMAP recomputation

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# From source (development)
git clone https://github.com/colejohnston/wikiscapes
cd wikiscapes
uv sync

# As a global tool
uv tool install .
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

---

## Quick Start

```bash
# 1. Initialize a wiki project in a new directory
mkdir my-wiki && cd my-wiki
wikiscapes init

# 2. Drop source files into raw/ and ingest them
wikiscapes ingest raw/paper.pdf
wikiscapes ingest raw/article.txt
wikiscapes ingest raw/notes.md

# 3. After 20+ articles, build the topographic map
wikiscapes evolve

# 4. Query your knowledge base
wikiscapes query "What are the key mechanisms of long-term potentiation?"

# 5. Visualize the map
wikiscapes map --format ascii
wikiscapes map --format html --output map.html
```

---

## Commands

### `wikiscapes init [path]`

Initialize a new wiki project. Creates:

```
my-wiki/
├── wikiscapes.toml   # project configuration
├── CLAUDE.md         # LLM governance schema
├── wiki/
│   └── index.md     # auto-maintained article catalog
├── raw/             # drop source files here
└── .wikiscapes/     # internal state (gitignored)
```

---

### `wikiscapes ingest <source>`

Ingest a raw source file into the wiki. Supported formats: `.txt`, `.md`, `.pdf`, `.html`.

```bash
wikiscapes ingest raw/paper.pdf
wikiscapes ingest raw/lecture-notes.txt --id "lecture-notes-week-3"
wikiscapes ingest raw/draft.md --dry-run   # preview without writing
```

**What happens:**
1. Extracts text from the source file
2. Loads `wiki/index.md` to check for existing related articles
3. Calls Claude Sonnet to generate a structured wiki article (or expand an existing one)
4. Writes `wiki/{article-id}.md` with YAML frontmatter
5. Computes an embedding and approximates the article's map position
6. Rebuilds `wiki/index.md`

---

### `wikiscapes query "<question>"`

Query the wiki using topographic routing.

```bash
wikiscapes query "What is the relationship between NMDA receptors and LTP?"
wikiscapes query "Summarize the evidence for the default mode network" --show-sources
wikiscapes query "What causes neuroplasticity?" --k 10      # retrieve more articles
wikiscapes query "Define synapse" --no-map                  # flat fallback (faster)
```

**6-phase routing algorithm:**
1. Embed the query (sentence-transformers or OpenAI)
2. Project to 2D map position (weighted centroid of nearest neighbors in embedding space — no UMAP recomputation)
3. Radius search on KDTree → candidate articles in that map neighborhood
4. Rerank candidates by full-dimension cosine similarity (corrects UMAP distortion)
5. Synthesize answer with Claude Sonnet (full body for top 2, Key Concepts only for articles 3+)
6. Log access to `.wikiscapes/access_log.jsonl` for plasticity tracking

The output includes the nearest cluster label so you can see where in the knowledge map the query landed.

---

### `wikiscapes evolve`

Re-layout the topographic map. Run after adding a significant number of articles or after many queries have shifted the access patterns.

```bash
wikiscapes evolve                  # runs only if thresholds are crossed
wikiscapes evolve --force          # run unconditionally
wikiscapes evolve --no-bridges     # skip bridge article generation
```

**What happens:**
1. Incrementally re-embeds new or stale articles (unchanged articles use cached embeddings)
2. Applies Homunculus weighting: high-importance articles are duplicated in the embedding matrix, causing UMAP to allocate more map area to frequently accessed regions
3. Runs full UMAP re-layout on all articles together (global coherence)
4. Runs HDBSCAN on 2D coordinates to find clusters
5. Labels changed clusters with Haiku (cached for unchanged clusters)
6. Updates all article frontmatter with new topology and neighbor connections
7. Detects confluence zones (where cluster KDEs overlap) using kernel density estimation
8. Generates bridge articles at qualifying confluence zones
9. Saves `MapState`, embeddings, and KDTree to `.wikiscapes/`
10. Rebuilds `wiki/index.md`

**Evolution thresholds** (when wikiscapes suggests running evolve):
- Article count changed by >20% since last layout
- >100 new queries logged since last layout

---

### `wikiscapes lint`

Run health checks across the wiki.

```bash
wikiscapes lint              # structural + topographic checks (free, no LLM)
wikiscapes lint --fix        # auto-repair fixable issues
wikiscapes lint --deep       # adds Haiku semantic validation (costs tokens)
```

**Structural checks (always, no LLM cost):**
- YAML parse errors in any article
- Missing source files referenced in frontmatter
- Broken `connections.local` references
- Duplicate article IDs
- Stale embedding versions
- Articles missing from `index.md`

**Topographic checks (always, no LLM cost):**
- Orphan articles (in HDBSCAN noise cluster)
- Map gap detection (10×10 grid scan for empty regions = missing knowledge domains)
- Cluster imbalance (any cluster >40% of all articles)
- High stub ratio (>20% of articles still `kind: stub`)
- Confluence zones without bridge articles

**Semantic checks (`--deep`, Haiku):**
- Spot-checks 5 random neighbor pairs ("are these actually related?")
- Detects articles whose source files are newer than `frontmatter.updated`

---

### `wikiscapes map`

Render the topographic map.

```bash
wikiscapes map                              # interactive HTML (default)
wikiscapes map --format html -o map.html
wikiscapes map --format png  -o map.png
wikiscapes map --format ascii               # terminal output
wikiscapes map --highlight "article-a,article-b"   # star-mark specific articles
```

**Interactive HTML map** (Plotly):
- Articles as dots, colored by cluster
- Dot size proportional to importance (access frequency × recency)
- Lines connecting local neighbor pairs
- Confluence zones as translucent ellipses
- Bridge articles as diamond markers
- Cluster labels at centroids
- Hover for title, cluster, access count, kind

**ASCII map** (terminal):
```
+--------------------------------------------------------------------------------+
|......Q...........Q.....Q..........Q.Q..Q..Q..Q....Q.Q..Q..Q..Q.Q..Q.Q.Q.Q.Q..|
|.....QQ..........QQQ...QQ.........QQ.QQQQ.QQ.QQQQ.QQ.QQ.QQ.QQ.Q.QQ.Q.Q.Q.Q.Q.|
|...N....N.......N..N..NNN.........NNNN.X.NNN.NNNN.NNNN.NN.NN.NN.N.NN.N.N.N.N.N|
+--------------------------------------------------------------------------------+

Legend:
  Q = Quantum Physics
  N = Neuroscience
  X = Confluence zone
```

---

### `wikiscapes status`

Print a summary of the current wiki state.

```bash
wikiscapes status
```

Output:

```
           Wikiscapes Status
  Articles:           87
  Stubs:              12
  Bridge articles:    4
  Layout version:     6
  Clusters:           8
  Confluence zones:   3
  Last evolved:       2026-04-13 09:15 UTC
  Embedding model:    all-MiniLM-L6-v2
```

---

### `wikiscapes add-raw <path>`

Copy a file into `raw/` with deduplication check.

```bash
wikiscapes add-raw ~/Downloads/paper.pdf
```

---

## Article Format

Every article in `wiki/` is a markdown file with YAML frontmatter managed by the engine. You can read articles directly but should not edit the `topo:`, `connections:`, or `access:` blocks manually.

```markdown
---
id: long-term-potentiation
title: "Long-Term Potentiation"
created: "2026-04-03T14:22:00Z"
updated: "2026-04-13T09:15:00Z"
sources:
  - raw/bliss-lomo-1973.pdf
  - raw/malenka-nicoll-1999.txt
kind: factual         # factual | synthesized | bridge | stub
abstraction_level: 0.15
importance: 0.74
topo:
  x: 0.42
  y: 0.71
  cluster_id: synaptic-plasticity
  cluster_label: Synaptic Plasticity
  embedding_version: "1"
connections:
  local:
    - nmda-receptors
    - hebbian-learning
    - calcium-signaling
  distant:
    - memory-consolidation
  confluence_of: []
access:
  count: 47
  last_accessed: "2026-04-13T08:00:00Z"
  query_contexts:
    - "NMDA receptor role in LTP"
    - "synaptic strengthening mechanisms"
---

# Long-Term Potentiation

Long-term potentiation (LTP) is a persistent strengthening of synapses...

## Mechanisms

...

## Key Concepts

- NMDA receptor activation
- Calcium influx
- AMPA receptor trafficking
- Hebbian plasticity
- Spike-timing dependent plasticity
```

---

## Configuration

`wikiscapes.toml` in your project root. All values have sensible defaults.

```toml
[wiki]
wiki_dir = "wiki"
raw_dir = "raw"
state_dir = ".wikiscapes"

[llm]
generation_model = "claude-sonnet-4-6"   # Ingest, query synthesis, bridge articles
fast_model = "claude-haiku-4-5"          # Cluster labels, lint checks, classification
max_context_tokens = 80000               # Token budget for query context window

[embeddings]
backend = "sentence-transformers"        # "sentence-transformers" (free) or "openai"
model = "all-MiniLM-L6-v2"              # 384-dim, fast, good quality
# For OpenAI: model = "text-embedding-3-small"
batch_size = 32

[umap]
n_neighbors = 15    # Increase to 25-30 for wikis with 150+ articles
min_dist = 0.05     # Lower than default → tighter clusters
metric = "cosine"
random_state = 42

[clustering]
min_cluster_size = 5   # Minimum articles to form a named cluster
min_samples = 3

[topology]
neighborhood_radius = 0.15         # Query search radius (0–1 normalized space)
neighborhood_radius_expanded = 0.25
k_fallback = 8                     # Fallback if radius search returns <3 results
k_local_connections = 5            # Stored neighbors per article
confluence_bandwidth = 0.12        # KDE bandwidth for confluence detection

[plasticity]
importance_weight_max = 3          # Max duplication for Homunculus weighting
evolve_article_change_threshold = 0.20
evolve_query_log_threshold = 100
access_log_flush_interval = 10     # Flush access counts every N queries
```

### Environment variables

Any config value can be overridden with an environment variable using the pattern `WIKISCAPES_SECTION__KEY` (double underscore for nesting):

```bash
export WIKISCAPES_LLM__GENERATION_MODEL="claude-opus-4-6"
export WIKISCAPES_EMBEDDINGS__BACKEND="openai"
export WIKISCAPES_UMAP__N_NEIGHBORS="25"
```

API keys are always environment variables:

```bash
export ANTHROPIC_API_KEY="..."     # Required for all LLM operations
export OPENAI_API_KEY="..."        # Required only if embeddings.backend = "openai"
```

---

## Model Selection and Cost

Wikiscapes routes operations to the cheapest model that can handle the task:

| Operation | Model | Why |
|-----------|-------|-----|
| Ingest article generation | Sonnet (full) | Synthesis, accuracy, encyclopedic structure |
| Query synthesis | Sonnet (full) | Multi-article reasoning, nuanced integration |
| Bridge article generation | Sonnet (full) | Cross-domain synthesis |
| Cluster label generation | Haiku (fast) | Simple classification from a list of titles |
| Lint neighbor validation | Haiku (fast) | Binary yes/no topographic check |
| Article kind/abstraction scoring | Haiku (fast) | Numeric rating from title + first paragraph |
| Structural + topographic lint | None | Pure Python, zero cost |

**Prompt caching** is applied to `INGEST_SYSTEM_PROMPT` and `SYNTHESIS_SYSTEM_PROMPT` — the longest system prompts, sent with every ingest/query call. This eliminates re-tokenization cost after the first call in each session.

**Token optimization strategies:**
- Embedding input truncated to `title + body[:2000]` (semantic fingerprint only)
- Query context: full body for top-2 articles, title + Key Concepts for articles 3+
- Access counts flushed to disk in batches (not per query)
- Cluster labels cached between evolve runs for unchanged clusters
- All structural and topographic lint checks run with zero API cost

---

## Architecture

```
wikiscapes/
├── models.py          # Pydantic models: Article, MapState, QueryResult, etc.
├── config.py          # wikiscapes.toml loader + env overrides
├── cli.py             # Typer CLI entry point
│
├── store/
│   ├── article_store.py   # Read/write articles with YAML frontmatter
│   ├── index.py           # wiki/index.md management
│   └── map_state.py       # MapState JSON, embeddings .npy, KDTree .pkl, access log
│
├── topo/
│   ├── embed.py           # sentence-transformers / OpenAI embedding backends
│   ├── layout.py          # UMAP layout, HDBSCAN clustering, Homunculus weighting
│   ├── spatial.py         # KDTree queries, query→map projection
│   └── confluence.py      # KDE-based confluence zone detection
│
├── llm/
│   ├── client.py          # Anthropic SDK wrapper (haiku/sonnet routing, prompt caching)
│   ├── prompts.py         # All prompt templates as versioned constants/factories
│   └── synthesis.py       # Response parsing: sources, gap detection, token budgeting
│
├── core/
│   ├── ingest.py          # Raw → wiki article pipeline
│   ├── query.py           # 6-phase topographic routing pipeline
│   ├── evolve.py          # UMAP re-layout, plasticity, bridge generation
│   └── lint.py            # Structural, topographic, and semantic health checks
│
└── viz/
    ├── plotly_map.py      # Interactive HTML topographic map
    └── static_map.py      # Matplotlib PNG + ASCII terminal map
```

### Internal state (`.wikiscapes/`)

```
.wikiscapes/
├── map_state.json     # Article positions, cluster assignments, confluence zones
├── embeddings.npy     # Cached embedding vectors (numpy binary)
├── embedding_ids.json # Article ID order matching embeddings.npy rows
├── kdtree.pkl         # Serialized scipy KDTree for fast neighborhood queries
└── access_log.jsonl   # Append-only query log (timestamp, query, articles_hit, coord)
```

This directory is gitignored by default (regenerated from `wiki/` content). Optionally commit `map_state.json` and `embeddings.npy` to pin a specific layout.

---

## CLAUDE.md Governance

`wikiscapes init` creates a `CLAUDE.md` in your project root. This file is the **schema governance document** — it tells any LLM working in this project how to write articles, handle duplicates, cite sources, and flag gaps. Keep it version-controlled. Edit it as your wiki's conventions evolve.

---

## Workflow Recommendations

**Starting a new wiki:**
1. Run `wikiscapes init`
2. Ingest 5–10 seed articles on your core topic
3. Run `wikiscapes query` — it will use flat retrieval until map is built
4. Once you have 20+ articles, run `wikiscapes evolve`
5. Check the map: `wikiscapes map --format ascii`

**Growing the wiki:**
- Ingest sources as you encounter them — `wikiscapes ingest raw/new-paper.pdf`
- Query freely — queries get better as the map develops
- Run `wikiscapes lint` periodically to catch gaps and broken links
- Run `wikiscapes evolve` when prompted (wikiscapes suggests it automatically)

**Using the map:**
- Open `map.html` in a browser after `wikiscapes map` to visually explore your knowledge
- Cluster labels tell you what each region is about
- Confluence zones (translucent ellipses) mark interdisciplinary regions
- Diamond markers are bridge articles synthesizing across domains
- Dot size encodes how frequently you've queried each article

**Keeping costs low:**
- Default `sentence-transformers` backend has zero embedding cost
- `wikiscapes lint` (no `--deep`) has zero API cost
- Use `wikiscapes query --no-map` for quick keyword-style lookups

---

## Brainscapes Concepts Implemented

| Book concept | Implementation |
|---|---|
| Retinotopic/tonotopic maps | UMAP 2D layout preserving semantic topology |
| Cortical homunculus | Importance-weighted embedding duplication before UMAP |
| Self-organized criticality | HDBSCAN cluster formation from layout, not imposed |
| Local lateral connectivity | `connections.local` links to k=5 nearest map neighbors |
| Confluence zones / cross-modal regions | KDE overlap detection between cluster boundaries |
| Activity-dependent plasticity | Access log → importance scores → periodic re-layout |
| Continuous gradients | `abstraction_level` field (0=concrete/factual, 1=abstract/meta) |
| Gap detection | Empty grid cells on the topographic map = missing knowledge domains |

---

## License

MIT
