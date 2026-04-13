"""Interactive HTML topographic map using Plotly."""

from __future__ import annotations

from pathlib import Path

from wikiscapes.models import Article, MapState

# ColorBrewer Set3 extended to 12+ distinct colors
_CLUSTER_COLORS = [
    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3",
    "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD",
    "#CCEBC5", "#FFED6F", "#A6CEE3", "#1F78B4", "#B2DF8A",
    "#33A02C",
]
_BRIDGE_COLOR = "#FFD700"  # Gold for bridge articles
_HIGHLIGHT_COLOR = "#FF4500"  # Orange-red for query result highlights


def render_interactive_map(
    map_state: MapState,
    articles: list[Article],
    output_path: Path,
    highlight_ids: list[str] | None = None,
    show_confluence: bool = True,
) -> None:
    """Render an interactive Plotly scatter map and write to output_path (HTML)."""
    import plotly.graph_objects as go  # type: ignore

    highlight_set = set(highlight_ids or [])
    id_to_article = {a.frontmatter.id: a for a in articles}

    # Assign a color to each unique cluster
    cluster_ids = sorted(set(
        pos.cluster_id for pos in map_state.articles.values()
    ))
    color_map = {cid: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}

    fig = go.Figure()

    # --- Connection lines ---
    edge_xs, edge_ys = [], []
    for article in articles:
        fm = article.frontmatter
        if fm.topo is None:
            continue
        for neighbor_id in fm.connections.local:
            n_pos = map_state.articles.get(neighbor_id)
            if n_pos is None:
                continue
            edge_xs += [fm.topo.x, n_pos.x, None]
            edge_ys += [fm.topo.y, n_pos.y, None]

    if edge_xs:
        fig.add_trace(go.Scatter(
            x=edge_xs, y=edge_ys,
            mode="lines",
            line={"color": "rgba(150,150,150,0.25)", "width": 0.8},
            hoverinfo="skip",
            showlegend=False,
            name="connections",
        ))

    # --- Confluence zones ---
    if show_confluence:
        for zone in map_state.confluence_zones:
            fig.add_shape(
                type="circle",
                x0=zone.centroid_x - zone.radius,
                y0=zone.centroid_y - zone.radius,
                x1=zone.centroid_x + zone.radius,
                y1=zone.centroid_y + zone.radius,
                fillcolor="rgba(255, 200, 0, 0.08)",
                line={"color": "rgba(255, 180, 0, 0.4)", "width": 1, "dash": "dot"},
            )

    # --- Article dots grouped by cluster ---
    for cluster_id in cluster_ids:
        cluster_articles = [
            a for a in articles
            if a.frontmatter.topo and a.frontmatter.topo.cluster_id == cluster_id
            and a.frontmatter.id not in highlight_set
            and a.frontmatter.kind != "bridge"
        ]
        if not cluster_articles:
            continue

        cluster_label = cluster_articles[0].frontmatter.topo.cluster_label if cluster_articles[0].frontmatter.topo else cluster_id

        xs = [a.frontmatter.topo.x for a in cluster_articles]  # type: ignore
        ys = [a.frontmatter.topo.y for a in cluster_articles]  # type: ignore
        sizes = [6 + 10 * a.frontmatter.importance for a in cluster_articles]
        hover = [
            f"<b>{a.frontmatter.title}</b><br>"
            f"cluster: {a.frontmatter.topo.cluster_label}<br>"  # type: ignore
            f"kind: {a.frontmatter.kind}<br>"
            f"access: {a.frontmatter.access.count}<br>"
            f"importance: {a.frontmatter.importance:.2f}"
            for a in cluster_articles
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker={"color": color_map[cluster_id], "size": sizes, "line": {"width": 0.5, "color": "white"}},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name=cluster_label,
        ))

    # --- Bridge articles ---
    bridge_articles = [a for a in articles if a.frontmatter.kind == "bridge" and a.frontmatter.topo]
    if bridge_articles:
        fig.add_trace(go.Scatter(
            x=[a.frontmatter.topo.x for a in bridge_articles],  # type: ignore
            y=[a.frontmatter.topo.y for a in bridge_articles],  # type: ignore
            mode="markers",
            marker={"symbol": "diamond", "color": _BRIDGE_COLOR, "size": 12, "line": {"width": 1, "color": "black"}},
            text=[f"<b>BRIDGE: {a.frontmatter.title}</b>" for a in bridge_articles],
            hovertemplate="%{text}<extra></extra>",
            name="Bridge Articles",
        ))

    # --- Highlighted articles (query results) ---
    highlighted = [a for a in articles if a.frontmatter.id in highlight_set and a.frontmatter.topo]
    if highlighted:
        fig.add_trace(go.Scatter(
            x=[a.frontmatter.topo.x for a in highlighted],  # type: ignore
            y=[a.frontmatter.topo.y for a in highlighted],  # type: ignore
            mode="markers",
            marker={"symbol": "star", "color": _HIGHLIGHT_COLOR, "size": 16, "line": {"width": 1, "color": "black"}},
            text=[f"<b>★ {a.frontmatter.title}</b>" for a in highlighted],
            hovertemplate="%{text}<extra></extra>",
            name="Query Results",
        ))

    # --- Cluster centroid labels ---
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
        fig.add_annotation(
            x=cx, y=cy, text=f"<b>{label}</b>",
            showarrow=False,
            font={"size": 11, "color": "rgba(40,40,40,0.8)"},
            bgcolor="rgba(255,255,255,0.6)",
            borderpad=2,
        )

    fig.update_layout(
        title="Wikiscapes Topographic Map",
        showlegend=True,
        hovermode="closest",
        xaxis={"visible": False, "range": [-0.05, 1.05]},
        yaxis={"visible": False, "range": [-0.05, 1.05]},
        plot_bgcolor="rgb(248, 248, 250)",
        paper_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"bgcolor": "rgba(255,255,255,0.8)", "bordercolor": "#ccc", "borderwidth": 1},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
