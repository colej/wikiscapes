"""Confluence zone detection — regions where distinct topic clusters overlap.

Algorithm:
1. For each cluster, fit a Gaussian KDE over article positions.
2. Find 2D regions where ≥2 cluster KDEs exceed a density threshold simultaneously.
3. Each such region becomes a ConfluenceZone.

Confluence zones are candidates for bridge article generation: cross-domain
synthesis articles that live at the conceptual intersection of two clusters.
"""

from __future__ import annotations

import hashlib

import numpy as np

from wikiscapes.models import Article, ConfluenceZone, MapState


def detect_confluence_zones(
    map_state: MapState,
    coords_norm: np.ndarray,
    cluster_labels: np.ndarray,
    ids: list[str],
    bandwidth: float = 0.12,
    min_clusters_meeting: int = 2,
    grid_resolution: int = 40,
    density_threshold: float = 0.3,
) -> list[ConfluenceZone]:
    """Detect regions where ≥ min_clusters_meeting cluster KDEs overlap.

    Args:
        coords_norm: normalized 2D coords, shape (N, 2)
        cluster_labels: HDBSCAN int labels, shape (N,)
        bandwidth: KDE bandwidth (controls spread of each cluster's density)
        grid_resolution: number of grid cells per axis for scanning
        density_threshold: fraction of max KDE density to count as "present"

    Returns list of ConfluenceZone objects.
    """
    from scipy.stats import gaussian_kde  # part of scipy

    unique_clusters = [c for c in set(cluster_labels.tolist()) if c != -1]
    if len(unique_clusters) < min_clusters_meeting:
        return []

    # Build KDE per cluster
    kde_per_cluster: dict[int, object] = {}
    for cid in unique_clusters:
        mask = cluster_labels == cid
        pts = coords_norm[mask]
        if len(pts) < 2:
            continue
        try:
            kde = gaussian_kde(pts.T, bw_method=bandwidth)
            kde_per_cluster[cid] = kde
        except Exception:
            continue

    if len(kde_per_cluster) < min_clusters_meeting:
        return []

    # Scan grid
    xs = np.linspace(0, 1, grid_resolution)
    ys = np.linspace(0, 1, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    # Evaluate each cluster KDE on the grid
    cluster_densities: dict[int, np.ndarray] = {}
    for cid, kde in kde_per_cluster.items():
        d = kde(grid_pts.T)  # type: ignore
        d_norm = d / (d.max() + 1e-12)  # normalize to [0, 1]
        cluster_densities[cid] = d_norm

    # Find grid cells where ≥ min_clusters_meeting clusters exceed threshold
    zones: list[ConfluenceZone] = []
    visited: set[int] = set()

    for cell_idx in range(len(grid_pts)):
        active_clusters = [
            cid for cid, d in cluster_densities.items() if d[cell_idx] >= density_threshold
        ]
        if len(active_clusters) < min_clusters_meeting:
            continue
        if cell_idx in visited:
            continue

        # Flood-fill connected cells with same active cluster set
        zone_cells = _flood_fill(
            cell_idx, grid_pts, cluster_densities, active_clusters,
            density_threshold, grid_resolution, visited
        )

        if not zone_cells:
            continue

        zone_coords = grid_pts[zone_cells]
        centroid = zone_coords.mean(axis=0)
        radius = float(np.linalg.norm(zone_coords - centroid, axis=1).max())

        # Compute density as fraction of cells in zone relative to grid area
        density = len(zone_cells) / (grid_resolution ** 2)

        cluster_id_strs = [
            _cluster_str(map_state, cid) for cid in active_clusters
        ]
        zone_id = _zone_id(cluster_id_strs, centroid)

        # Preserve existing bridge_article_id if this zone already exists
        existing_bridge = _find_existing_bridge(map_state, zone_id)

        zones.append(ConfluenceZone(
            id=zone_id,
            centroid_x=float(centroid[0]),
            centroid_y=float(centroid[1]),
            clusters=cluster_id_strs,
            radius=radius,
            density=density,
            bridge_article_id=existing_bridge,
        ))

    return zones


def should_generate_bridge(
    zone: ConfluenceZone,
    map_state: MapState,
    min_articles_per_cluster: int = 3,
    min_zone_density: float = 0.3,
) -> bool:
    """Return True if this zone meets criteria for bridge article generation."""
    if zone.bridge_article_id is not None:
        return False  # Already has a bridge
    if zone.density < min_zone_density:
        return False
    for cluster_id in zone.clusters:
        count = len(map_state.clusters.get(cluster_id, []))
        if count < min_articles_per_cluster:
            return False
    return True


def find_bridge_articles(
    zone: ConfluenceZone,
    articles: list[Article],
    coords_norm: np.ndarray,
    ids: list[str],
    k_per_cluster: int = 3,
) -> list[Article]:
    """Return the k articles closest to the zone centroid from each contributing cluster."""
    centroid = np.array([zone.centroid_x, zone.centroid_y])
    id_to_article = {a.frontmatter.id: a for a in articles}
    result: list[Article] = []

    for cluster_id in zone.clusters:
        # Find articles in this cluster
        cluster_indices = [
            i for i, aid in enumerate(ids)
            if aid in id_to_article
            and id_to_article[aid].frontmatter.topo is not None
            and id_to_article[aid].frontmatter.topo.cluster_id == cluster_id
        ]
        if not cluster_indices:
            continue

        cluster_coords = coords_norm[cluster_indices]
        dists = np.linalg.norm(cluster_coords - centroid, axis=1)
        nearest = np.argsort(dists)[:k_per_cluster]
        for ni in nearest:
            aid = ids[cluster_indices[ni]]
            if aid in id_to_article:
                result.append(id_to_article[aid])

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flood_fill(
    start: int,
    grid_pts: np.ndarray,
    cluster_densities: dict[int, np.ndarray],
    required_clusters: list[int],
    threshold: float,
    resolution: int,
    visited: set[int],
) -> list[int]:
    """BFS flood-fill over grid cells where all required clusters exceed threshold."""
    queue = [start]
    zone_cells: list[int] = []

    while queue:
        cell = queue.pop()
        if cell in visited:
            continue
        active = all(cluster_densities[cid][cell] >= threshold for cid in required_clusters)
        if not active:
            continue
        visited.add(cell)
        zone_cells.append(cell)

        # 4-connected grid neighbors
        row, col = divmod(cell, resolution)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < resolution and 0 <= nc < resolution:
                neighbor = nr * resolution + nc
                if neighbor not in visited:
                    queue.append(neighbor)

    return zone_cells


def _cluster_str(map_state: MapState, cluster_int: int) -> str:
    """Convert int cluster label to string cluster_id used in MapState."""
    # MapState.clusters is keyed by cluster_id strings (e.g. "quantum-physics")
    # During evolve, we build a reverse mapping; here we fall back to str(int)
    for cid in map_state.clusters:
        if cid == str(cluster_int):
            return cid
    return str(cluster_int)


def _zone_id(cluster_ids: list[str], centroid: np.ndarray) -> str:
    key = "_".join(sorted(cluster_ids)) + f"_{centroid[0]:.2f}_{centroid[1]:.2f}"
    return "zone-" + hashlib.md5(key.encode()).hexdigest()[:8]


def _find_existing_bridge(map_state: MapState, zone_id: str) -> str | None:
    for zone in map_state.confluence_zones:
        if zone.id == zone_id:
            return zone.bridge_article_id
    return None
