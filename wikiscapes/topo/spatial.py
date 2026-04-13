"""KDTree-based spatial index for topographic neighborhood queries.

All coordinates are in normalized [0, 1] × [0, 1] space.

Key function: project_query_to_map()
  Projects a new query into the existing 2D map using the weighted centroid of
  its k-nearest neighbors in the original high-dimensional embedding space.
  This avoids UMAP recomputation per query.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def build_kdtree(coords_normalized: np.ndarray) -> KDTree:
    """Build a KDTree from normalized 2D coordinates."""
    return KDTree(coords_normalized)


def k_nearest_neighbors(
    tree: KDTree,
    ids: list[str],
    query_id: str,
    k: int = 5,
) -> list[str]:
    """Return ids of k nearest articles to query_id by 2D Euclidean distance."""
    if query_id not in ids:
        return []
    idx = ids.index(query_id)
    point = tree.data[idx]
    # k+1 because the point itself is always nearest
    dists, indices = tree.query(point, k=min(k + 1, len(ids)))
    return [ids[i] for i in indices if ids[i] != query_id][:k]


def query_neighborhood(
    tree: KDTree,
    ids: list[str],
    query_point: np.ndarray,
    radius: float = 0.15,
    k_fallback: int = 8,
    min_results: int = 3,
) -> list[str]:
    """Return article ids within radius of query_point.

    If fewer than min_results articles fall within radius, expands to k_fallback
    nearest neighbors to ensure a useful result set.
    """
    indices = tree.query_ball_point(query_point, r=radius)
    if len(indices) >= min_results:
        return [ids[i] for i in indices]

    # Fallback: k-nearest
    k = min(k_fallback, len(ids))
    _, fallback_indices = tree.query(query_point, k=k)
    if isinstance(fallback_indices, (int, np.integer)):
        fallback_indices = [fallback_indices]
    return [ids[i] for i in fallback_indices]


def project_query_to_map(
    query_embedding: np.ndarray,
    article_embeddings: np.ndarray,
    article_coords: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Project a query into the 2D map without re-running UMAP.

    Finds the k nearest articles to the query in the full embedding space,
    then returns the distance-weighted centroid of their 2D coordinates.
    This is an approximation but accurate enough for neighborhood routing.

    Args:
        query_embedding: 1D array of shape (embedding_dim,)
        article_embeddings: 2D array of shape (n_articles, embedding_dim)
        article_coords: 2D array of shape (n_articles, 2), normalized [0,1]
        k: number of neighbors to use for centroid estimation

    Returns:
        1D array of shape (2,) — estimated map position
    """
    # Cosine similarity = dot product when embeddings are L2-normalized
    # article_embeddings from sentence-transformers are already normalized
    similarities = article_embeddings @ query_embedding

    k_actual = min(k, len(similarities))
    top_indices = np.argpartition(similarities, -k_actual)[-k_actual:]

    top_sims = similarities[top_indices]
    # Shift to positive weights (similarities can range -1 to 1)
    weights = top_sims - top_sims.min() + 1e-6
    weights = weights / weights.sum()

    weighted_coord = (article_coords[top_indices] * weights[:, None]).sum(axis=0)
    return weighted_coord


def rerank_by_embedding_similarity(
    candidate_ids: list[str],
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    all_ids: list[str],
    top_k: int = 6,
) -> list[str]:
    """Rerank candidate article ids by full-dimension cosine similarity.

    Corrects for UMAP projection distortion in sparse map regions.
    """
    id_to_idx = {eid: i for i, eid in enumerate(all_ids)}
    candidates = [(eid, id_to_idx[eid]) for eid in candidate_ids if eid in id_to_idx]
    if not candidates:
        return []

    indices = [idx for _, idx in candidates]
    sims = all_embeddings[indices] @ query_embedding
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return [eid for (eid, _), _ in ranked[:top_k]]
