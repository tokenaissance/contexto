"""Tests for the local clusterer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from contexto_hermes.local.clustering import Clusterer
from contexto_hermes.local.mindmap_types import (
    ClusterNode,
    ConversationItem,
    MindmapConfig,
    StoreState,
    StoreStats,
)


def _item(i: int, embedding: list[float], content: str | None = None) -> ConversationItem:
    return ConversationItem(
        id=f"item-{i}",
        role="assistant",
        content=content or f"item-{i} content",
        embedding=embedding,
    )


def _normalize(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 0 else v


def _empty_state() -> StoreState:
    return StoreState()


class TestRebuildPolicy:
    def test_rebuild_when_total_under_100(self):
        # All items below 100 should ALWAYS trigger rebuild (`new_total < 100`),
        # regardless of inserts_since_rebuild.
        clusterer = Clusterer(MindmapConfig(rebuild_interval=10))
        state = _empty_state()
        state.stats.inserts_since_rebuild = 5  # well below interval
        items = [_item(i, [1.0, 0.0]) for i in range(5)]
        state = clusterer.add(state, items)
        assert state.stats.inserts_since_rebuild == 0  # rebuild reset counter
        assert state.stats.total_items == 5

    def test_rebuild_when_threshold_hit_after_100(self):
        clusterer = Clusterer(MindmapConfig(rebuild_interval=20))
        state = _empty_state()
        # Seed with 100 items (forces rebuild)
        seed = [_item(i, _normalize([1.0 + i * 0.01, i * 0.005])) for i in range(100)]
        state = clusterer.add(state, seed)
        assert state.stats.total_items == 100
        assert state.stats.inserts_since_rebuild == 0

        # Add 10 more — should NOT rebuild (total >= 100, increment < interval)
        more = [_item(100 + i, _normalize([0.0, 1.0 + i * 0.01])) for i in range(10)]
        state = clusterer.add(state, more)
        assert state.stats.total_items == 110
        assert state.stats.inserts_since_rebuild == 10

        # Add 10 more — incremental sum hits 20, triggers rebuild.
        more2 = [_item(110 + i, _normalize([1.0, 1.0 + i * 0.01])) for i in range(10)]
        state = clusterer.add(state, more2)
        assert state.stats.total_items == 120
        assert state.stats.inserts_since_rebuild == 0


class TestBuildShapes:
    def test_zero_items_empty_root(self):
        clusterer = Clusterer(MindmapConfig())
        state = clusterer.add(_empty_state(), [])
        assert state.root is None

    def test_one_item_root_with_single_leaf(self):
        clusterer = Clusterer(MindmapConfig())
        state = clusterer.add(_empty_state(), [_item(0, [1.0, 0.0])])
        assert state.root is not None
        assert state.root.id == "root"
        assert state.root.item_count == 1
        assert len(state.root.children) == 1
        assert state.root.children[0].items[0].id == "item-0"

    def test_two_distant_items_get_split(self):
        # Two clusters of orthogonal vectors should be separated.
        clusterer = Clusterer(MindmapConfig(similarity_threshold=0.5, max_depth=3))
        items = [
            _item(0, [1.0, 0.0]),
            _item(1, [1.0, 0.01]),
            _item(2, [0.0, 1.0]),
            _item(3, [0.01, 1.0]),
        ]
        state = clusterer.add(_empty_state(), items)
        assert state.root is not None
        # Root should have >= 2 children (the two clusters)
        assert len(state.root.children) >= 2
        assert state.root.item_count == 4


class TestIncrementalInsert:
    def test_incremental_path_updates_centroid(self):
        clusterer = Clusterer(MindmapConfig(rebuild_interval=200))
        state = _empty_state()
        # Seed with 100 items so threshold is met (no rebuild on incremental adds).
        seed = [_item(i, _normalize([1.0, i * 0.01])) for i in range(100)]
        state = clusterer.add(state, seed)
        before_total = state.stats.total_items
        before_inserts = state.stats.inserts_since_rebuild  # 0

        # Add one more item (incremental path).
        state = clusterer.add(state, [_item(100, _normalize([1.0, 0.5]))])
        assert state.stats.total_items == before_total + 1
        assert state.stats.inserts_since_rebuild == before_inserts + 1
        assert state.root is not None
        assert state.root.item_count == 101


class TestStatsInvariants:
    def test_total_items_matches_item_count(self):
        clusterer = Clusterer(MindmapConfig())
        items = [_item(i, _normalize([1.0, i * 0.01])) for i in range(20)]
        state = clusterer.add(_empty_state(), items)
        assert state.root is not None
        assert state.stats.total_items == state.root.item_count == 20

    def test_total_clusters_positive(self):
        clusterer = Clusterer(MindmapConfig())
        items = [_item(i, _normalize([1.0, i * 0.01])) for i in range(20)]
        state = clusterer.add(_empty_state(), items)
        assert state.stats.total_clusters >= 1


def _collect_cluster_ids(node: ClusterNode) -> list[str]:
    ids = [node.id]
    for child in node.children:
        ids.extend(_collect_cluster_ids(child))
    return ids


class TestClusterIdUniquenessAfterReload:
    """Reviewer-reproduced scenario: incremental insert against a loaded state
    must not reuse `cluster-N` ids that already exist in the tree.
    """

    def test_incremental_insert_after_reload_does_not_collide(self):
        # Phase 1 — build state with a fresh clusterer; capture the resulting IDs.
        builder = Clusterer(MindmapConfig(rebuild_interval=1000))
        seed = [_item(i, _normalize([1.0, i * 0.01])) for i in range(100)]
        state = builder.add(_empty_state(), seed)
        first_ids = _collect_cluster_ids(state.root)
        assert state.root is not None
        assert len(set(first_ids)) == len(first_ids), "seed build should already be unique"

        # Phase 2 — simulate a process restart: throw away the builder, mint a
        # FRESH Clusterer (which starts its counter at 1) against the loaded
        # state. Forced incremental path: new_total >= 100 AND well under
        # rebuild_interval.
        reloaded = Clusterer(MindmapConfig(rebuild_interval=1000))
        # Force the incremental path: new_total stays >= 100 and far under
        # rebuild_interval.
        extra = [_item(200 + i, _normalize([0.0, 1.0 + i * 0.01])) for i in range(3)]
        state = reloaded.add(state, extra)

        all_ids = _collect_cluster_ids(state.root)
        assert len(set(all_ids)) == len(all_ids), (
            f"duplicate cluster ids after reload+incremental: {sorted(all_ids)}"
        )
        # Specifically: every new id should be strictly larger than every preexisting id.
        preexisting_max = max(
            int(cid.split("-", 1)[1]) for cid in first_ids if cid.startswith("cluster-")
        )
        new_ids = [cid for cid in all_ids if cid not in set(first_ids) and cid.startswith("cluster-")]
        for cid in new_ids:
            assert int(cid.split("-", 1)[1]) > preexisting_max

    def test_seed_counter_idempotent(self):
        # Two consecutive incremental adds against the same reloaded state
        # should still produce unique ids (the seed step is idempotent).
        builder = Clusterer(MindmapConfig(rebuild_interval=1000))
        state = builder.add(_empty_state(), [
            _item(i, _normalize([1.0, i * 0.01])) for i in range(100)
        ])

        reloaded = Clusterer(MindmapConfig(rebuild_interval=1000))
        state = reloaded.add(state, [_item(200, _normalize([0.0, 1.0]))])
        state = reloaded.add(state, [_item(201, _normalize([0.0, 1.0]))])
        ids = _collect_cluster_ids(state.root)
        assert len(set(ids)) == len(ids)
