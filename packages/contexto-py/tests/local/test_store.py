"""Tests for the local mindmap store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contexto_hermes.local.mindmap_types import (
    ClusterNode,
    ConversationItem,
    StoreState,
    StoreStats,
)
from contexto_hermes.local.store import Store, SCHEMA_VERSION


def _sample_state() -> StoreState:
    item = ConversationItem(
        id="item-1",
        role="assistant",
        content="hello world",
        embedding=[0.1, 0.2, 0.3],
        timestamp="2026-05-23T00:00:00.000Z",
        metadata={"source": "summary", "status": "complete"},
    )
    cluster = ClusterNode(
        id="cluster-1",
        label="deployment errors",
        centroid=[0.1, 0.2, 0.3],
        children=[],
        items=[item],
        depth=1,
        item_count=1,
    )
    root = ClusterNode(
        id="root",
        label="Knowledge",
        centroid=[0.1, 0.2, 0.3],
        children=[cluster],
        items=[],
        depth=0,
        item_count=1,
    )
    return StoreState(
        version=SCHEMA_VERSION,
        config_snapshot={"similarity_threshold": 0.65},
        root=root,
        stats=StoreStats(total_items=1, total_clusters=2, inserts_since_rebuild=0),
    )


class TestLoadMissingFile:
    def test_returns_empty_state(self, tmp_path):
        store = Store(str(tmp_path / "missing.json"))
        state = store.load()
        assert state.version == SCHEMA_VERSION
        assert state.root is None
        assert state.stats == StoreStats()


class TestRoundTrip:
    def test_save_then_load(self, tmp_path):
        path = tmp_path / "mindmap.json"
        store = Store(str(path))
        store.save(_sample_state())
        loaded = store.load()
        assert loaded.version == SCHEMA_VERSION
        assert loaded.stats.total_items == 1
        assert loaded.stats.total_clusters == 2
        assert loaded.root is not None
        assert loaded.root.id == "root"
        assert loaded.root.children[0].label == "deployment errors"
        assert loaded.root.children[0].items[0].content == "hello world"
        assert loaded.root.children[0].items[0].metadata["source"] == "summary"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "mindmap.json"
        store = Store(str(path))
        store.save(_sample_state())
        assert path.exists()

    def test_atomic_write_uses_tmp_then_rename(self, tmp_path):
        # The .tmp file should not exist after a successful save.
        path = tmp_path / "mindmap.json"
        store = Store(str(path))
        store.save(_sample_state())
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()


class TestQuarantine:
    def test_unparseable_json_is_quarantined(self, tmp_path):
        path = tmp_path / "mindmap.json"
        path.write_text("not json {{", encoding="utf-8")
        store = Store(str(path))
        state = store.load()
        assert state.root is None
        # Original file moved aside; subsequent saves still work.
        assert any(p.name.startswith("mindmap.json.corrupted-") for p in tmp_path.iterdir())
        # File no longer at the original path.
        assert not path.exists()

    def test_unknown_version_is_quarantined(self, tmp_path):
        path = tmp_path / "mindmap.json"
        path.write_text(json.dumps({"version": 99, "root": None, "stats": {}}), encoding="utf-8")
        store = Store(str(path))
        state = store.load()
        assert state.version == SCHEMA_VERSION
        assert state.root is None
        assert any(p.name.startswith("mindmap.json.corrupted-") for p in tmp_path.iterdir())

    def test_non_dict_stats_is_quarantined(self, tmp_path):
        # F9: `stats` must be an object. A string / list would previously raise
        # AttributeError on `.get(...)` before quarantine. Verify it now
        # quarantines cleanly and returns a fresh state.
        path = tmp_path / "mindmap.json"
        path.write_text(
            json.dumps({"version": SCHEMA_VERSION, "stats": "broken", "root": None}),
            encoding="utf-8",
        )
        store = Store(str(path))
        state = store.load()
        assert state.root is None
        assert state.stats == StoreStats()
        assert any(p.name.startswith("mindmap.json.corrupted-") for p in tmp_path.iterdir())

    def test_top_level_array_is_quarantined(self, tmp_path):
        path = tmp_path / "mindmap.json"
        path.write_text("[]", encoding="utf-8")
        store = Store(str(path))
        state = store.load()
        assert state.root is None
        assert any(p.name.startswith("mindmap.json.corrupted-") for p in tmp_path.iterdir())

    def test_save_after_quarantine_writes_fresh_state(self, tmp_path):
        path = tmp_path / "mindmap.json"
        path.write_text("garbage", encoding="utf-8")
        store = Store(str(path))
        store.load()  # quarantine
        store.save(_sample_state())
        # Original path now contains the fresh state.
        assert path.exists()
        loaded = store.load()
        assert loaded.stats.total_items == 1


class TestSchema:
    def test_disk_has_version_field(self, tmp_path):
        path = tmp_path / "mindmap.json"
        Store(str(path)).save(_sample_state())
        raw = json.loads(path.read_text())
        assert raw["version"] == SCHEMA_VERSION
        assert "config_snapshot" in raw
        assert "stats" in raw
        assert "root" in raw

    def test_empty_state_persists_with_none_root(self, tmp_path):
        path = tmp_path / "mindmap.json"
        store = Store(str(path))
        store.save(StoreState(version=SCHEMA_VERSION))
        raw = json.loads(path.read_text())
        assert raw["root"] is None
