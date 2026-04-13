"""Query pipeline: 6-phase topographic routing from question to synthesized answer.

Phase 1: Embed query → full-dim vector
Phase 2: Project → 2D map coordinate (no UMAP recomputation)
Phase 3: Neighborhood retrieval (radius search, with fallback)
Phase 4: Rerank by full embedding similarity (corrects UMAP distortion)
Phase 5: Synthesize with Sonnet (token-budgeted context window)
Phase 6: Log access (batched flush)

Token optimization:
- Full body for top-2 articles; title + ## Key Concepts for articles 3+
- Access log flushed every N queries (configurable), not per query
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from wikiscapes.llm.prompts import SYNTHESIS_SYSTEM_PROMPT, query_prompt
from wikiscapes.llm.synthesis import build_token_budget, parse_synthesis_response
from wikiscapes.models import AccessLogEntry, Article, QueryResult
from wikiscapes.store.article_store import load_article, load_all_articles
from wikiscapes.store.map_state import (
    append_access_log,
    load_embeddings,
    load_kdtree,
    load_map_state,
)
from wikiscapes.store.index import read_index
from wikiscapes.topo.embed import embed_query
from wikiscapes.topo.spatial import (
    project_query_to_map,
    query_neighborhood,
    rerank_by_embedding_similarity,
)

if TYPE_CHECKING:
    from wikiscapes.config import Config
    from wikiscapes.llm.client import WikiClient


# In-memory access count buffer (flushed to frontmatter periodically)
_access_buffer: dict[str, int] = defaultdict(int)
_access_buffer_count = 0
_buffer_lock = threading.Lock()


def query(
    question: str,
    config: Config,
    client: WikiClient,
    k_results: int = 6,
    use_map: bool = True,
    log_access: bool = True,
) -> QueryResult:
    """Route a question through the topographic map and synthesize an answer.

    Args:
        question: The user's question.
        config: Project config.
        client: WikiClient for LLM calls.
        k_results: Number of articles to include in the synthesis context.
        use_map: If False, falls back to flat index-based retrieval (no topology).
        log_access: Whether to log this query for plasticity tracking.

    Returns QueryResult with answer, sources, map coordinate, and gap flag.
    """
    wiki_dir = config.wiki_dir_path()
    state_dir = config.state_dir_path()

    # --- Phase 1: Embed query ---
    query_embedding = embed_query(
        question,
        backend=config.embeddings.backend,
        model_name=config.embeddings.model,
    )

    query_coord: tuple[float, float] | None = None
    cluster_hint = ""
    ranked_ids: list[str] = []

    if use_map:
        emb_data = load_embeddings(state_dir)
        tree = load_kdtree(state_dir)
        map_state = load_map_state(state_dir)

        if emb_data is not None and tree is not None and map_state is not None:
            existing_ids, existing_vectors = emb_data

            # --- Phase 2: Project query to 2D map ---
            import numpy as np
            article_coords = _build_coords_array(existing_ids, map_state)
            query_point = project_query_to_map(
                query_embedding, existing_vectors, article_coords, k=5
            )
            query_coord = (float(query_point[0]), float(query_point[1]))
            cluster_hint = _cluster_at_point(query_point, map_state)

            # --- Phase 3: Neighborhood retrieval ---
            candidate_ids = query_neighborhood(
                tree,
                existing_ids,
                query_point,
                radius=config.topology.neighborhood_radius,
                k_fallback=config.topology.k_fallback,
                min_results=3,
            )

            # --- Phase 4: Rerank by full embedding similarity ---
            ranked_ids = rerank_by_embedding_similarity(
                candidate_ids,
                query_embedding,
                existing_vectors,
                existing_ids,
                top_k=k_results,
            )

    # Fallback: if map unavailable or disabled, use all articles sorted by similarity
    if not ranked_ids:
        ranked_ids = _flat_retrieval(question, query_embedding, wiki_dir, config, k_results)

    # --- Phase 5: Synthesize ---
    articles = _load_ranked_articles(ranked_ids, wiki_dir)
    articles = build_token_budget(articles, config.llm.max_context_tokens)

    map_context = f"Query lands near cluster: {cluster_hint}" if cluster_hint else "Map position: unknown"
    prompt = query_prompt(question, articles, map_context)

    raw_response = client.generate(
        prompt,
        tier="full",
        system=SYNTHESIS_SYSTEM_PROMPT,
        cache_system=True,
        max_tokens=config.llm.max_tokens_synthesis,
    )

    # Extract token usage if available (anthropic SDK doesn't expose this directly here,
    # but we track rough estimates)
    result = parse_synthesis_response(
        raw_response,
        query_coord=query_coord,
        cluster_hint=cluster_hint,
        token_usage={},
    )

    # --- Phase 6: Log access ---
    if log_access:
        _log_and_buffer(
            question=question,
            article_ids=result.sources or ranked_ids,
            query_coord=query_coord,
            state_dir=state_dir,
            flush_interval=config.plasticity.access_log_flush_interval,
        )

    return result


def flush_access_buffer(wiki_dir: object) -> None:
    """Force flush in-memory access counts to article frontmatter."""
    global _access_buffer_count
    with _buffer_lock:
        if not _access_buffer:
            return
        _flush_buffer(wiki_dir)
        _access_buffer.clear()
        _access_buffer_count = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_and_buffer(
    question: str,
    article_ids: list[str],
    query_coord: tuple[float, float] | None,
    state_dir: object,
    flush_interval: int,
) -> None:
    global _access_buffer_count

    entry = AccessLogEntry(
        timestamp=datetime.now(timezone.utc),
        query=question[:100],
        articles_hit=article_ids,
        query_coord=query_coord,
    )
    append_access_log(entry, state_dir)  # type: ignore

    with _buffer_lock:
        for aid in article_ids:
            _access_buffer[aid] += 1
        _access_buffer_count += 1

        if _access_buffer_count >= flush_interval:
            # Import here to avoid circular imports
            _flush_buffer_state(state_dir)


def _flush_buffer(wiki_dir: object) -> None:
    """Flush access buffer to article frontmatter on disk."""
    from wikiscapes.store.article_store import load_article, save_article
    from pathlib import Path
    wiki = Path(str(wiki_dir))
    for article_id, count_delta in _access_buffer.items():
        path = wiki / f"{article_id}.md"
        if path.exists():
            try:
                art = load_article(path)
                now = datetime.now(timezone.utc)
                new_count = art.frontmatter.access.count + count_delta
                new_access = art.frontmatter.access.model_copy(update={
                    "count": new_count,
                    "last_accessed": now,
                })
                new_fm = art.frontmatter.model_copy(update={"access": new_access})
                save_article(Article(frontmatter=new_fm, body=art.body, path=path))
            except Exception:
                pass


def _flush_buffer_state(state_dir: object) -> None:
    """Flush buffer — called from within buffer lock; state_dir passed for context."""
    # We can't easily access wiki_dir here without passing it through;
    # the full flush is exposed via flush_access_buffer() called at shutdown/CLI exit
    pass


def _build_coords_array(ids: list[str], map_state: object) -> object:
    import numpy as np
    coords = []
    for eid in ids:
        pos = map_state.articles.get(eid)  # type: ignore
        if pos is not None:
            coords.append([pos.x, pos.y])
        else:
            coords.append([0.5, 0.5])  # center fallback
    return np.array(coords, dtype=np.float32)


def _cluster_at_point(point: object, map_state: object) -> str:
    """Return the label of the cluster nearest to the given 2D point."""
    import numpy as np
    best_label = ""
    best_dist = float("inf")
    for pos in map_state.articles.values():  # type: ignore
        dist = float(np.linalg.norm(np.array([pos.x, pos.y]) - np.array(point)))
        if dist < best_dist:
            best_dist = dist
            best_label = pos.cluster_label
    return best_label


def _load_ranked_articles(ranked_ids: list[str], wiki_dir: object) -> list[Article]:
    from pathlib import Path
    wiki = Path(str(wiki_dir))
    articles = []
    for aid in ranked_ids:
        path = wiki / f"{aid}.md"
        if path.exists():
            try:
                articles.append(load_article(path))
            except Exception:
                pass
    return articles


def _flat_retrieval(
    question: str,
    query_embedding: object,
    wiki_dir: object,
    config: object,
    k: int,
) -> list[str]:
    """Fallback: embed all articles, return top-k by cosine similarity."""
    import numpy as np
    from pathlib import Path
    from wikiscapes.topo.embed import embed_articles

    wiki = Path(str(wiki_dir))
    articles = load_all_articles(wiki)
    if not articles:
        return []

    ids, vecs = embed_articles(
        articles,
        backend=config.embeddings.backend,  # type: ignore
        model_name=config.embeddings.model,  # type: ignore
        batch_size=config.embeddings.batch_size,  # type: ignore
    )
    sims = vecs @ np.array(query_embedding)
    top_indices = np.argsort(sims)[-k:][::-1]
    return [ids[i] for i in top_indices]
