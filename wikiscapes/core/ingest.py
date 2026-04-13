"""Ingest pipeline: raw source file → placed wiki article.

Steps:
1. Detect file type and extract text
2. Auto-generate article_id (or accept override)
3. Load index.md for LLM context (lazy — caller may pass cached index)
4. Call Sonnet to generate the wiki article body
5. Check for UPDATE: directive (expand existing article instead of creating new)
6. Scaffold ArticleFrontmatter and write wiki/{article_id}.md
7. Generate embedding → project to approximate map position
8. Rebuild KDTree → update connections.local
9. Rebuild index.md
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from wikiscapes.llm.prompts import INGEST_SYSTEM_PROMPT, ingest_prompt
from wikiscapes.llm.synthesis import extract_update_directive
from wikiscapes.models import Article, ArticleFrontmatter
from wikiscapes.store.article_store import (
    load_all_articles,
    load_article,
    save_article,
    slugify,
    unique_article_id,
)
from wikiscapes.store.index import read_index, rebuild_index
from wikiscapes.store.map_state import (
    load_embeddings,
    load_kdtree,
    load_map_state,
    save_embeddings,
    save_kdtree,
)
from wikiscapes.topo.embed import embed_articles, embed_query
from wikiscapes.topo.spatial import build_kdtree, k_nearest_neighbors, project_query_to_map

if TYPE_CHECKING:
    from wikiscapes.config import Config
    from wikiscapes.llm.client import WikiClient


def ingest(
    source_path: Path,
    config: Config,
    client: WikiClient,
    article_id: str | None = None,
    dry_run: bool = False,
    cached_index: str | None = None,
) -> Article:
    """Ingest a raw source file into the wiki.

    Args:
        source_path: Path to the raw source (txt, md, pdf, html).
        config: Project config.
        client: WikiClient for LLM calls.
        article_id: Override the auto-generated slug.
        dry_run: If True, print the generated article but do not write to disk.
        cached_index: Pre-loaded index.md string (avoids re-read in batch runs).

    Returns the created or updated Article.
    """
    wiki_dir = config.wiki_dir_path()
    state_dir = config.state_dir_path()

    # Step 1: Extract text
    source_text = _extract_text(source_path)

    # Step 2: Resolve article_id
    base_id = article_id or slugify(source_path.stem)
    resolved_id = unique_article_id(base_id, wiki_dir)

    # Step 3: Load index (lazy)
    index_text = cached_index if cached_index is not None else read_index(wiki_dir)

    # Step 4: Generate article via Sonnet
    prompt = ingest_prompt(source_text, index_text, resolved_id)
    raw_response = client.generate(
        prompt,
        tier="full",
        system=INGEST_SYSTEM_PROMPT,
        cache_system=True,
        max_tokens=config.llm.max_tokens_ingest,
    )

    # Step 5: Check for UPDATE directive
    existing_id, body = extract_update_directive(raw_response)

    if existing_id:
        existing_path = wiki_dir / f"{existing_id}.md"
        if existing_path.exists():
            existing = load_article(existing_path)
            now = datetime.now(timezone.utc)
            updated_fm = existing.frontmatter.model_copy(update={
                "updated": now,
                "sources": list(set(existing.frontmatter.sources + [str(source_path)])),
            })
            article = Article(frontmatter=updated_fm, body=body, path=existing_path)
            if not dry_run:
                save_article(article)
            return article

    # Step 6: Scaffold frontmatter and write article
    now = datetime.now(timezone.utc)
    fm = ArticleFrontmatter(
        id=resolved_id,
        title=_extract_title(body) or resolved_id.replace("-", " ").title(),
        created=now,
        updated=now,
        sources=[str(source_path)],
        kind="stub",
        abstraction_level=0.5,
    )
    article_path = wiki_dir / f"{resolved_id}.md"
    article = Article(frontmatter=fm, body=body, path=article_path)

    if dry_run:
        return article

    wiki_dir.mkdir(parents=True, exist_ok=True)
    save_article(article)

    # Step 7: Embedding + approximate map placement
    article = _place_on_map(article, config, state_dir)

    # Step 8–9: Rebuild KDTree and index
    all_articles = load_all_articles(wiki_dir)
    rebuild_index(all_articles, wiki_dir)

    return article


def _place_on_map(article: Article, config: Config, state_dir: Path) -> Article:
    """Generate embedding and assign approximate 2D topo position."""
    emb_data = load_embeddings(state_dir)
    if emb_data is None:
        # No map exists yet; leave topo=None until first evolve
        return article

    existing_ids, existing_vectors = emb_data

    # Embed the new article
    new_ids, new_vecs = embed_articles(
        [article],
        backend=config.embeddings.backend,
        model_name=config.embeddings.model,
        batch_size=1,
    )
    new_vec = new_vecs[0]

    # Load existing 2D coordinates from map state
    map_state = load_map_state(state_dir)
    if map_state is None or not map_state.articles:
        return article

    article_coords = _build_coords_array(existing_ids, map_state)
    if article_coords is None:
        return article

    # Project to 2D
    approx_coord = project_query_to_map(new_vec, existing_vectors, article_coords, k=5)

    # Find cluster at this position
    cluster_id, cluster_label = _nearest_cluster(approx_coord, map_state)

    # Update article frontmatter with approximate position
    from wikiscapes.models import TopoPosition
    from wikiscapes.store.article_store import save_article

    topo = TopoPosition(
        x=float(approx_coord[0]),
        y=float(approx_coord[1]),
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        embedding_version="1",
    )
    updated_fm = article.frontmatter.model_copy(update={"topo": topo})
    article = Article(frontmatter=updated_fm, body=article.body, path=article.path)
    save_article(article)

    # Rebuild KDTree with new article included
    all_ids = existing_ids + [article.frontmatter.id]
    import numpy as np
    all_vecs = np.vstack([existing_vectors, new_vec[None]])
    save_embeddings(all_ids, all_vecs, state_dir)

    # Update connections.local using KDTree
    all_coords = _build_full_coords_array(all_ids, map_state, approx_coord, article.frontmatter.id)
    if all_coords is not None:
        import numpy as np
        tree = build_kdtree(all_coords)
        save_kdtree(tree, state_dir)
        neighbors = k_nearest_neighbors(tree, all_ids, article.frontmatter.id, k=config.topology.k_local_connections)
        conn = article.frontmatter.connections.model_copy(update={"local": neighbors})
        updated_fm2 = article.frontmatter.model_copy(update={"connections": conn})
        article = Article(frontmatter=updated_fm2, body=article.body, path=article.path)
        save_article(article)

    return article


def _extract_text(path: Path) -> str:
    """Extract plain text from txt, md, pdf, or html files."""
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"[Page {i}]\n{text}")
        return "\n\n".join(pages)

    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    # Fallback: try reading as text
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_title(body: str) -> str | None:
    """Extract the first # heading from the article body."""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _build_coords_array(ids: list[str], map_state: object) -> object | None:
    import numpy as np
    coords = []
    for eid in ids:
        pos = map_state.articles.get(eid)  # type: ignore
        if pos is None:
            return None
        coords.append([pos.x, pos.y])
    if not coords:
        return None
    return np.array(coords, dtype=np.float32)


def _build_full_coords_array(
    all_ids: list[str],
    map_state: object,
    new_coord: object,
    new_id: str,
) -> object | None:
    import numpy as np
    coords = []
    for eid in all_ids:
        if eid == new_id:
            coords.append(list(new_coord))
        else:
            pos = map_state.articles.get(eid)  # type: ignore
            if pos is None:
                return None
            coords.append([pos.x, pos.y])
    return np.array(coords, dtype=np.float32)


def _nearest_cluster(coord: object, map_state: object) -> tuple[str, str]:
    """Return the cluster_id and label nearest to the given 2D coordinate."""
    import numpy as np
    best_id = "unplaced"
    best_label = "Unplaced"
    best_dist = float("inf")

    for article_id, pos in map_state.articles.items():  # type: ignore
        dist = np.linalg.norm(np.array([pos.x, pos.y]) - np.array(coord))
        if dist < best_dist:
            best_dist = dist
            best_id = pos.cluster_id
            best_label = pos.cluster_label

    return best_id, best_label
