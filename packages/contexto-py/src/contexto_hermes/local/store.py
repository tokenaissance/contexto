"""JSON load/save for the local mindmap state.

- Atomic write via `<path>.tmp` + `os.replace`.
- Parent directory autocreated.
- Corrupt or wrong-version files are renamed to `<path>.corrupted-<unix-millis>`
  and a fresh empty `StoreState` is returned. Single-writer assumption (v1).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .mindmap_types import (
    ClusterNode,
    ConversationItem,
    StoreState,
    StoreStats,
)

logger = logging.getLogger("plugins.context_engine.contexto")

SCHEMA_VERSION = 1


def _node_from_dict(d: Any) -> ClusterNode | None:
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError("ClusterNode must be a dict")
    children = [_node_from_dict(c) for c in d.get("children", [])]
    items = [_item_from_dict(i) for i in d.get("items", [])]
    return ClusterNode(
        id=str(d["id"]),
        label=str(d.get("label", "")),
        centroid=list(d.get("centroid") or []),
        children=[c for c in children if c is not None],
        items=items,
        depth=int(d.get("depth", 0)),
        item_count=int(d.get("item_count", 0)),
    )


def _item_from_dict(d: Any) -> ConversationItem:
    if not isinstance(d, dict):
        raise ValueError("ConversationItem must be a dict")
    return ConversationItem(
        id=str(d["id"]),
        role=str(d.get("role", "")),
        content=str(d.get("content", "")),
        embedding=list(d.get("embedding") or []),
        timestamp=d.get("timestamp"),
        metadata=dict(d.get("metadata") or {}),
    )


def _state_to_dict(state: StoreState) -> dict[str, Any]:
    return {
        "version": state.version,
        "config_snapshot": state.config_snapshot,
        "stats": asdict(state.stats),
        "root": _node_to_dict(state.root) if state.root is not None else None,
    }


def _node_to_dict(node: ClusterNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "centroid": node.centroid,
        "children": [_node_to_dict(c) for c in node.children],
        "items": [_item_to_dict(i) for i in node.items],
        "depth": node.depth,
        "item_count": node.item_count,
    }


def _item_to_dict(item: ConversationItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "role": item.role,
        "content": item.content,
        "embedding": item.embedding,
        "timestamp": item.timestamp,
        "metadata": item.metadata,
    }


class Store:
    """File-backed mindmap store. Lazy-loads on first call to load()."""

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> StoreState:
        """Read state from disk. Quarantine corrupt/incompatible files."""
        if not self._path.exists():
            return _empty_state()

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            self._quarantine(reason=f"unreadable: {exc}")
            return _empty_state()

        if not isinstance(data, dict):
            self._quarantine(reason="top-level JSON is not an object")
            return _empty_state()

        version = data.get("version")
        if version != SCHEMA_VERSION:
            self._quarantine(reason=f"unknown schema version: {version!r}")
            return _empty_state()

        # Validate `stats` BEFORE indexing it — a non-dict truthy value (e.g.
        # `"garbage"` or `[1,2]`) would have raised AttributeError on
        # `stats_dict.get(...)`. Quarantine in that case.
        raw_stats = data.get("stats")
        if raw_stats is not None and not isinstance(raw_stats, dict):
            self._quarantine(reason=f"`stats` must be an object, got {type(raw_stats).__name__}")
            return _empty_state()
        stats_dict: dict[str, Any] = raw_stats or {}

        try:
            root = _node_from_dict(data.get("root"))
            stats = StoreStats(
                total_items=int(stats_dict.get("total_items", 0)),
                total_clusters=int(stats_dict.get("total_clusters", 0)),
                inserts_since_rebuild=int(stats_dict.get("inserts_since_rebuild", 0)),
            )
            config_snapshot = data.get("config_snapshot") or {}
            if not isinstance(config_snapshot, dict):
                config_snapshot = {}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            self._quarantine(reason=f"schema mismatch: {exc}")
            return _empty_state()

        return StoreState(
            version=SCHEMA_VERSION,
            config_snapshot=config_snapshot,
            root=root,
            stats=stats,
        )

    def save(self, state: StoreState) -> None:
        """Atomic write to `self._path`. Raises on I/O failure."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = _state_to_dict(state)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)

    def _quarantine(self, *, reason: str) -> None:
        """Rename the bad file aside; fresh state returned to caller."""
        millis = int(time.time() * 1000)
        backup = self._path.with_suffix(self._path.suffix + f".corrupted-{millis}")
        try:
            os.replace(self._path, backup)
            logger.error(
                "[contexto:local] mindmap store quarantined (%s): renamed %s → %s",
                reason, self._path, backup,
            )
        except OSError as exc:
            logger.error(
                "[contexto:local] mindmap store quarantine FAILED (%s); leaving file in place: %s",
                reason, exc,
            )


def _empty_state() -> StoreState:
    return StoreState(version=SCHEMA_VERSION, config_snapshot={}, root=None, stats=StoreStats())


__all__ = ["Store", "SCHEMA_VERSION"]
