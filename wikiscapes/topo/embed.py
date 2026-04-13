"""Embedding generation for articles and queries.

Backends:
  - sentence-transformers (default, free, local)
  - openai (text-embedding-3-small, better cross-domain, costs tokens)

Embedding input: title + "\\n\\n" + body[:2000]
Only the relevant semantic fingerprint is needed; tail content rarely shifts direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from wikiscapes.models import Article

if TYPE_CHECKING:
    pass

_EMBED_TRUNCATE = 2000


def _build_text(article: Article) -> str:
    return article.frontmatter.title + "\n\n" + article.body[:_EMBED_TRUNCATE]


def embed_articles(
    articles: list[Article],
    backend: Literal["sentence-transformers", "openai"],
    model_name: str,
    batch_size: int = 32,
    existing_ids: list[str] | None = None,
    existing_vectors: np.ndarray | None = None,
    current_embedding_version: str = "1",
) -> tuple[list[str], np.ndarray]:
    """Generate embeddings for articles.

    Incremental: if existing_ids/existing_vectors are provided, only articles
    whose id is absent or whose embedding_version differs are re-embedded.

    Returns (ids, embedding_matrix) ordered by articles list.
    """
    # Build lookup of existing embeddings
    existing: dict[str, np.ndarray] = {}
    if existing_ids is not None and existing_vectors is not None:
        for i, eid in enumerate(existing_ids):
            existing[eid] = existing_vectors[i]

    to_embed: list[tuple[int, Article]] = []
    result_vectors: list[np.ndarray | None] = [None] * len(articles)

    for idx, article in enumerate(articles):
        fm = article.frontmatter
        cached = existing.get(fm.id)
        if cached is not None and (
            fm.topo is None or fm.topo.embedding_version == current_embedding_version
        ):
            result_vectors[idx] = cached
        else:
            to_embed.append((idx, article))

    if to_embed:
        texts = [_build_text(a) for _, a in to_embed]
        if backend == "sentence-transformers":
            new_vecs = _embed_st(texts, model_name, batch_size)
        else:
            new_vecs = _embed_openai(texts, model_name, batch_size)
        for (idx, _), vec in zip(to_embed, new_vecs):
            result_vectors[idx] = vec

    ids = [a.frontmatter.id for a in articles]
    matrix = np.vstack([v for v in result_vectors if v is not None])
    return ids, matrix


def embed_query(
    query: str,
    backend: Literal["sentence-transformers", "openai"],
    model_name: str,
) -> np.ndarray:
    """Return a single embedding vector for a query string."""
    if backend == "sentence-transformers":
        vecs = _embed_st([query], model_name, batch_size=1)
    else:
        vecs = _embed_openai([query], model_name, batch_size=1)
    return vecs[0]


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _embed_st(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = _get_st_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


_st_model_cache: dict[str, object] = {}


def _get_st_model(model_name: str) -> object:
    from sentence_transformers import SentenceTransformer  # type: ignore

    if model_name not in _st_model_cache:
        _st_model_cache[model_name] = SentenceTransformer(model_name)
    return _st_model_cache[model_name]


def _embed_openai(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    import os

    import openai  # type: ignore

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        for item in response.data:
            all_vectors.append(item.embedding)
    return np.array(all_vectors, dtype=np.float32)
