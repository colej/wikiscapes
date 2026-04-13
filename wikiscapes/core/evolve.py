"""Map evolution pipeline: re-layout, re-cluster, plasticity, bridge generation.

Steps:
1. Incremental re-embed (only new/stale articles)
2. Apply Homunculus importance weighting
3. Full UMAP re-layout (all articles together)
4. HDBSCAN cluster assignment
5. Re-label changed clusters via Haiku (cached for unchanged)
6. Rebuild KDTree → update all article topo + connections.local
7. Recompute importance scores
8. Detect confluence zones → generate bridge articles
9. Save MapState, embeddings, KDTree
10. Rebuild wiki/index.md
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from wikiscapes.models import Article, EvolveResult, MapState, TopoPosition
from wikiscapes.store.article_store import load_all_articles, save_article
from wikiscapes.store.index import rebuild_index
from wikiscapes.store.map_state import (
    count_access_log_since,
    load_embeddings,
    load_map_state,
    save_embeddings,
    save_kdtree,
    save_map_state,
)
from wikiscapes.topo.confluence import (
    detect_confluence_zones,
    find_bridge_articles,
    should_generate_bridge,
)
from wikiscapes.topo.embed import embed_articles
from wikiscapes.topo.layout import (
    assign_clusters,
    compute_layout,
    deduplicate_weighted_coords,
    label_clusters,
    umap_params_dict,
    weight_embeddings_by_importance,
)
from wikiscapes.topo.spatial import build_kdtree, k_nearest_neighbors

if TYPE_CHECKING:
    from wikiscapes.config import Config
    from wikiscapes.llm.client import WikiClient


def evolve(
    config: Config,
    client: WikiClient,
    force: bool = False,
    generate_bridges: bool = True,
) -> EvolveResult:
    """Re-layout the topographic map and apply plasticity updates.

    Returns an EvolveResult summary.
    """
    start = time.time()
    wiki_dir = config.wiki_dir_path()
    state_dir = config.state_dir_path()

    articles = load_all_articles(wiki_dir)
    if not articles:
        return EvolveResult(
            articles_re_embedded=0,
            clusters_changed=0,
            bridges_generated=0,
            layout_version=0,
            duration_seconds=0.0,
        )

    old_state = load_map_state(state_dir)

    # Step 1: Incremental re-embed
    emb_data = load_embeddings(state_dir)
    existing_ids, existing_vecs = emb_data if emb_data else ([], np.empty((0, 0)))

    current_version = "1"
    ids, embeddings = embed_articles(
        articles,
        backend=config.embeddings.backend,
        model_name=config.embeddings.model,
        batch_size=config.embeddings.batch_size,
        existing_ids=existing_ids if existing_ids else None,
        existing_vectors=existing_vecs if len(existing_vecs) else None,
        current_embedding_version=current_version,
    )
    re_embedded = _count_re_embedded(articles, existing_ids, current_version)

    # Step 2: Homunculus weighting
    importances = {a.frontmatter.id: a.frontmatter.importance for a in articles}
    expanded_ids, expanded_embeddings = weight_embeddings_by_importance(
        ids, embeddings, importances, max_weight=config.plasticity.importance_weight_max
    )

    # Step 3: UMAP layout
    umap_kwargs = umap_params_dict(
        config.umap.n_neighbors,
        config.umap.min_dist,
        config.umap.metric,
        config.umap.random_state,
    )
    _, expanded_coords_norm = compute_layout(expanded_ids, expanded_embeddings, **umap_kwargs)

    # Collapse duplicates back to single positions
    coords_norm = deduplicate_weighted_coords(expanded_ids, expanded_coords_norm, ids)

    # Step 4: HDBSCAN clustering
    cluster_labels = assign_clusters(
        coords_norm,
        min_cluster_size=config.clustering.min_cluster_size,
        min_samples=config.clustering.min_samples,
    )

    # Step 5: Label clusters (Haiku, cached for unchanged membership)
    old_labels, old_membership = _extract_old_cluster_info(old_state, articles)
    cluster_label_map = label_clusters(
        cluster_labels, articles, client,
        existing_labels=old_labels,
        existing_membership=old_membership,
    )
    clusters_changed = _count_changed_clusters(old_labels, cluster_label_map)

    # Step 6: Build cluster_id → article mapping
    cluster_id_map: dict[str, str] = {}  # int_label → cluster_id string
    cluster_groups: dict[str, list[str]] = {}
    for article, label_int in zip(articles, cluster_labels):
        label_str = str(int(label_int))
        label_text = cluster_label_map.get(int(label_int), "Unplaced")
        cluster_id = _slugify_cluster(label_text, label_int)
        cluster_id_map[label_str] = cluster_id
        cluster_groups.setdefault(cluster_id, []).append(article.frontmatter.id)

    # Step 7: Build new MapState
    new_layout_version = (old_state.layout_version + 1) if old_state else 1
    topo_positions: dict[str, TopoPosition] = {}
    for article, label_int, coord in zip(articles, cluster_labels, coords_norm):
        label_str = str(int(label_int))
        cluster_id = cluster_id_map.get(label_str, "unplaced")
        cluster_label_text = cluster_label_map.get(int(label_int), "Unplaced")
        topo_positions[article.frontmatter.id] = TopoPosition(
            x=float(coord[0]),
            y=float(coord[1]),
            cluster_id=cluster_id,
            cluster_label=cluster_label_text,
            embedding_version=current_version,
        )

    # Step 8: Detect confluence zones
    now = datetime.now(timezone.utc)
    temp_state = MapState(
        articles=topo_positions,
        clusters=cluster_groups,
        confluence_zones=[],
        layout_version=new_layout_version,
        article_count_at_layout=len(articles),
        embedding_model=config.embeddings.model,
        umap_params=umap_kwargs,
        created=old_state.created if old_state else now,
        last_evolved=now,
    )

    confluence_zones = detect_confluence_zones(
        temp_state,
        coords_norm,
        cluster_labels,
        ids,
        bandwidth=config.topology.confluence_bandwidth,
    )

    # Step 9: Generate bridge articles
    bridges_generated = 0
    if generate_bridges:
        for zone in confluence_zones:
            if should_generate_bridge(zone, temp_state):
                bridge_articles = find_bridge_articles(zone, articles, coords_norm, ids)
                bridge_id = _generate_bridge_article(
                    zone, bridge_articles, config, client, cluster_label_map,
                    cluster_id_map, articles, cluster_labels
                )
                if bridge_id:
                    zone.bridge_article_id = bridge_id
                    bridges_generated += 1

    # Update MapState with confluence zones
    final_state = temp_state.model_copy(update={"confluence_zones": confluence_zones})

    # Step 10: Update all article frontmatter (topo + connections.local + importance)
    max_access = max((a.frontmatter.access.count for a in articles), default=1) or 1
    tree = build_kdtree(coords_norm)

    for article, coord in zip(articles, coords_norm):
        aid = article.frontmatter.id
        topo = topo_positions[aid]
        neighbors = k_nearest_neighbors(tree, ids, aid, k=config.topology.k_local_connections)
        importance = _compute_importance(article, max_access, now)

        conn = article.frontmatter.connections.model_copy(update={"local": neighbors})
        updated_fm = article.frontmatter.model_copy(update={
            "topo": topo,
            "connections": conn,
            "importance": importance,
            "updated": now,
        })
        updated_article = Article(
            frontmatter=updated_fm, body=article.body, path=article.path
        )
        save_article(updated_article)

    # Step 11: Persist state
    save_embeddings(ids, embeddings, state_dir)
    save_kdtree(tree, state_dir)
    save_map_state(final_state, state_dir)

    # Reload articles (includes any newly generated bridge articles)
    all_articles = load_all_articles(wiki_dir)
    rebuild_index(all_articles, wiki_dir)

    return EvolveResult(
        articles_re_embedded=re_embedded,
        clusters_changed=clusters_changed,
        bridges_generated=bridges_generated,
        layout_version=new_layout_version,
        duration_seconds=time.time() - start,
    )


def should_suggest_evolve(
    config: Config,
) -> tuple[bool, str]:
    """Check if evolution thresholds have been crossed.

    Returns (should_evolve, reason_message).
    """
    wiki_dir = config.wiki_dir_path()
    state_dir = config.state_dir_path()

    articles = load_all_articles(wiki_dir)
    map_state = load_map_state(state_dir)

    if map_state is None:
        return True, "No map state exists — run evolve to build initial topology."

    current_count = len(articles)
    prior_count = map_state.article_count_at_layout
    if prior_count > 0:
        change_ratio = abs(current_count - prior_count) / prior_count
        if change_ratio >= config.plasticity.evolve_article_change_threshold:
            return True, (
                f"Article count changed by {change_ratio:.0%} since last layout "
                f"({prior_count} → {current_count})."
            )

    new_entries = count_access_log_since(state_dir, map_state.last_evolved)
    if new_entries >= config.plasticity.evolve_query_log_threshold:
        return True, (
            f"{new_entries} queries logged since last evolve "
            f"(threshold: {config.plasticity.evolve_query_log_threshold})."
        )

    return False, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_re_embedded(
    articles: list[Article], existing_ids: list[str], current_version: str
) -> int:
    existing_set = set(existing_ids)
    count = 0
    for a in articles:
        if a.frontmatter.id not in existing_set:
            count += 1
        elif a.frontmatter.topo and a.frontmatter.topo.embedding_version != current_version:
            count += 1
    return count


def _extract_old_cluster_info(
    old_state: MapState | None,
    articles: list[Article],
) -> tuple[dict[int, str], dict[int, set[str]]]:
    if old_state is None:
        return {}, {}
    # Build reverse: cluster_id_str → cluster_int (approximate — use label index)
    labels: dict[int, str] = {}
    membership: dict[int, set[str]] = {}
    # We store clusters by label string in MapState; reconstruct integer keys heuristically
    for i, (cid, aids) in enumerate(old_state.clusters.items()):
        if cid == "unplaced":
            continue
        labels[i] = cid
        membership[i] = set(aids)
    return labels, membership


def _count_changed_clusters(old: dict[int, str], new: dict[int, str]) -> int:
    changed = 0
    for k in new:
        if old.get(k) != new.get(k):
            changed += 1
    return changed


def _slugify_cluster(label: str, label_int: int) -> str:
    import re
    if label_int == -1:
        return "unplaced"
    slug = label.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-") or f"cluster-{label_int}"


def _compute_importance(
    article: Article, max_access: int, now: datetime
) -> float:
    """Sigmoid-scaled importance: access frequency × recency weight."""
    import math
    count = article.frontmatter.access.count
    freq_score = count / max_access if max_access > 0 else 0.0

    last = article.frontmatter.access.last_accessed
    if last:
        days_ago = max(0, (now - last).days)
        recency = 1.0 / (1.0 + math.log1p(days_ago))
    else:
        recency = 0.1

    raw = freq_score * recency
    # Sigmoid to squash to (0, 1)
    return 1.0 / (1.0 + math.exp(-10 * (raw - 0.5)))


def _generate_bridge_article(
    zone: object,
    bridge_articles: list[Article],
    config: Config,
    client: WikiClient,
    cluster_label_map: dict[int, str],
    cluster_id_map: dict[str, str],
    all_articles: list[Article],
    cluster_labels: object,
) -> str | None:
    """Generate a bridge article for the given confluence zone."""
    from wikiscapes.llm.prompts import BRIDGE_ARTICLE_SYSTEM_PROMPT, bridge_article_prompt
    from wikiscapes.store.article_store import save_article, unique_article_id
    from wikiscapes.models import ArticleFrontmatter, ArticleConnections

    if len(bridge_articles) < 2:
        return None

    wiki_dir = config.wiki_dir_path()
    zone_clusters = zone.clusters  # type: ignore
    cluster_labels_text = [
        _cluster_label_from_id(cid, cluster_label_map, cluster_id_map)
        for cid in zone_clusters[:2]
    ]
    cluster_a = cluster_labels_text[0] if cluster_labels_text else "Domain A"
    cluster_b = cluster_labels_text[1] if len(cluster_labels_text) > 1 else "Domain B"

    bridge_id_base = f"bridge-{'-'.join(sorted(zone_clusters[:2]))}"
    bridge_id = unique_article_id(bridge_id_base, wiki_dir)

    prompt = bridge_article_prompt(cluster_a, cluster_b, bridge_articles, bridge_id)
    body = client.generate(
        prompt,
        tier="full",
        system=BRIDGE_ARTICLE_SYSTEM_PROMPT,
        max_tokens=config.llm.max_tokens_ingest,
    )

    now = datetime.now(timezone.utc)
    from wikiscapes.models import TopoPosition
    topo = TopoPosition(
        x=zone.centroid_x,  # type: ignore
        y=zone.centroid_y,  # type: ignore
        cluster_id="bridge",
        cluster_label=f"Bridge: {cluster_a} ↔ {cluster_b}",
        embedding_version="1",
    )
    conn = ArticleConnections(confluence_of=zone_clusters)  # type: ignore
    fm = ArticleFrontmatter(
        id=bridge_id,
        title=f"Bridge: {cluster_a} and {cluster_b}",
        created=now,
        updated=now,
        sources=[a.frontmatter.id for a in bridge_articles],
        topo=topo,
        connections=conn,
        kind="bridge",
        abstraction_level=0.7,
    )
    article_path = wiki_dir / f"{bridge_id}.md"
    bridge_article = Article(frontmatter=fm, body=body, path=article_path)
    save_article(bridge_article)
    return bridge_id


def _cluster_label_from_id(
    cluster_id: str,
    cluster_label_map: dict[int, str],
    cluster_id_map: dict[str, str],
) -> str:
    # Reverse lookup: cluster_id → label text
    for int_label, cid in cluster_id_map.items():
        if cid == cluster_id:
            return cluster_label_map.get(int(int_label.split("-")[-1]) if "-" in int_label else int(int_label), cluster_id)
    return cluster_id
