"""Persistence layer for MapState, embeddings, KDTree, and access log."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wikiscapes.models import AccessLogEntry, MapState

if TYPE_CHECKING:
    from scipy.spatial import KDTree


_MAP_STATE_FILE = "map_state.json"
_EMBEDDINGS_FILE = "embeddings.npy"
_EMBEDDING_IDS_FILE = "embedding_ids.json"
_KDTREE_FILE = "kdtree.pkl"
_ACCESS_LOG_FILE = "access_log.jsonl"


def load_map_state(state_dir: Path) -> MapState | None:
    path = state_dir / _MAP_STATE_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MapState.model_validate(data)


def save_map_state(state: MapState, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _MAP_STATE_FILE
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def load_embeddings(state_dir: Path) -> tuple[list[str], np.ndarray] | None:
    ids_path = state_dir / _EMBEDDING_IDS_FILE
    emb_path = state_dir / _EMBEDDINGS_FILE
    if not ids_path.exists() or not emb_path.exists():
        return None
    ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    vectors = np.load(str(emb_path))
    return ids, vectors


def save_embeddings(ids: list[str], vectors: np.ndarray, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / _EMBEDDING_IDS_FILE).write_text(
        json.dumps(ids, indent=2), encoding="utf-8"
    )
    np.save(str(state_dir / _EMBEDDINGS_FILE), vectors)


def load_kdtree(state_dir: Path) -> KDTree | None:
    path = state_dir / _KDTREE_FILE
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_kdtree(tree: KDTree, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / _KDTREE_FILE, "wb") as f:
        pickle.dump(tree, f)


def append_access_log(entry: AccessLogEntry, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / _ACCESS_LOG_FILE
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def read_access_log(state_dir: Path) -> list[AccessLogEntry]:
    log_path = state_dir / _ACCESS_LOG_FILE
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(AccessLogEntry.model_validate_json(line))
    return entries


def count_access_log_since(state_dir: Path, since: datetime) -> int:
    """Count access log entries newer than `since` (for plasticity thresholds)."""
    entries = read_access_log(state_dir)
    since_utc = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since
    return sum(1 for e in entries if e.timestamp > since_utc)
