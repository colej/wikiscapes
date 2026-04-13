"""Static PNG and ASCII terminal map renderers."""

from __future__ import annotations

from pathlib import Path

from wikiscapes.models import Article, MapState

_ASCII_WIDTH = 80
_ASCII_HEIGHT = 30


def render_static_map(
    map_state: MapState,
    articles: list[Article],
    output_path: Path,
) -> None:
    """Render a publication-quality PNG via matplotlib."""
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as mpatches  # type: ignore
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_facecolor("#f8f8fa")
    fig.patch.set_facecolor("white")

    # Assign colors by cluster index
    cluster_ids = sorted(set(pos.cluster_id for pos in map_state.articles.values()))
    cmap = plt.cm.get_cmap("tab20", max(len(cluster_ids), 1))
    color_map = {cid: cmap(i) for i, cid in enumerate(cluster_ids)}

    # Draw connections
    for article in articles:
        fm = article.frontmatter
        if fm.topo is None:
            continue
        for neighbor_id in fm.connections.local:
            n_pos = map_state.articles.get(neighbor_id)
            if n_pos is None:
                continue
            ax.plot(
                [fm.topo.x, n_pos.x], [fm.topo.y, n_pos.y],
                color="gray", alpha=0.15, linewidth=0.5, zorder=1,
            )

    # Draw confluence zones
    for zone in map_state.confluence_zones:
        circle = plt.Circle(
            (zone.centroid_x, zone.centroid_y),
            zone.radius,
            color="gold", alpha=0.08, zorder=2,
        )
        ax.add_patch(circle)

    # Draw articles
    for article in articles:
        fm = article.frontmatter
        if fm.topo is None:
            continue
        color = color_map.get(fm.topo.cluster_id, "gray")
        size = 40 + 80 * fm.importance
        marker = "D" if fm.kind == "bridge" else "o"
        ax.scatter(
            fm.topo.x, fm.topo.y,
            s=size, c=[color], marker=marker,  # type: ignore
            edgecolors="white", linewidths=0.5, zorder=3, alpha=0.85,
        )

    # Cluster centroid labels
    for cluster_id in cluster_ids:
        positions = [
            map_state.articles[aid]
            for aid in map_state.clusters.get(cluster_id, [])
            if aid in map_state.articles
        ]
        if not positions:
            continue
        cx = sum(p.x for p in positions) / len(positions)
        cy = sum(p.y for p in positions) / len(positions)
        label = positions[0].cluster_label
        ax.text(
            cx, cy, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            zorder=4,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[cid], label=map_state.articles[aids[0]].cluster_label
                       if (aids := map_state.clusters.get(cid, [])) and aids[0] in map_state.articles
                       else cid)
        for cid in cluster_ids
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches, loc="lower right",
            fontsize=8, framealpha=0.8,
        )

    ax.set_title("Wikiscapes Topographic Map", fontsize=14, pad=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_ascii_map(
    map_state: MapState,
    articles: list[Article],
    width: int = _ASCII_WIDTH,
    height: int = _ASCII_HEIGHT,
) -> str:
    """Render a text-mode topographic map for terminal output.

    Cell contents:
      '.' = empty
      First letter of cluster label = 1–3 articles in cell
      Digit (1–9) = count of articles (capped at 9)
      'X' = confluence zone centroid
    """
    # Initialize grid
    grid: list[list[str]] = [["." for _ in range(width)] for _ in range(height)]
    cell_counts: dict[tuple[int, int], int] = {}
    cell_labels: dict[tuple[int, int], str] = {}

    for article in articles:
        fm = article.frontmatter
        if fm.topo is None:
            continue
        col = min(int(fm.topo.x * width), width - 1)
        row = min(int((1 - fm.topo.y) * height), height - 1)  # flip y for terminal
        cell = (row, col)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        if fm.topo.cluster_label and cell not in cell_labels:
            cell_labels[cell] = fm.topo.cluster_label[0].upper()

    for (row, col), count in cell_counts.items():
        if count <= 3:
            grid[row][col] = cell_labels.get((row, col), "·")
        else:
            grid[row][col] = str(min(count, 9))

    # Mark confluence zone centroids
    for zone in map_state.confluence_zones:
        col = min(int(zone.centroid_x * width), width - 1)
        row = min(int((1 - zone.centroid_y) * height), height - 1)
        grid[row][col] = "X"

    # Build output
    border = "+" + "-" * width + "+"
    lines = [border]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append(border)

    # Legend
    cluster_ids = sorted(set(pos.cluster_id for pos in map_state.articles.values()))
    legend_items = []
    for cid in cluster_ids:
        aids = map_state.clusters.get(cid, [])
        if aids and aids[0] in map_state.articles:
            label = map_state.articles[aids[0]].cluster_label
            initial = label[0].upper()
            legend_items.append(f"{initial} = {label}")

    if legend_items:
        lines.append("")
        lines.append("Legend:")
        for item in legend_items:
            lines.append(f"  {item}")
        lines.append("  X = Confluence zone")
        lines.append(f"  . = Empty region | Digits = article count (max 9)")

    return "\n".join(lines)
