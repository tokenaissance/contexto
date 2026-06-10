"""Tests for beam_search."""

from __future__ import annotations

import numpy as np

from contexto_hermes.local.mindmap_types import (
    ClusterNode,
    ConversationItem,
    MindmapConfig,
)
from contexto_hermes.local.retrieval import beam_search


def _normalize(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 0 else v


def _item(i: int, embedding: list[float], **md) -> ConversationItem:
    return ConversationItem(
        id=f"item-{i}",
        role="assistant",
        content=f"content {i}",
        embedding=embedding,
        metadata=md,
    )


def _node(
    id: str, label: str, centroid: list[float],
    children=None, items=None, depth=1, item_count=None,
) -> ClusterNode:
    items = items or []
    children = children or []
    return ClusterNode(
        id=id, label=label, centroid=centroid,
        children=children, items=items, depth=depth,
        item_count=item_count if item_count is not None else len(items),
    )


def _root_with(children: list[ClusterNode], centroid: list[float] | None = None) -> ClusterNode:
    total = sum(c.item_count for c in children)
    return _node("root", "Knowledge", centroid or [0.5, 0.5], children=children, depth=0, item_count=total)


class TestBasics:
    def test_empty_tree_no_terminals_returns_no_items(self):
        # root with no children — falls through to root as terminal (empty itself)
        root = _root_with([])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, [1.0, 0.0], cfg, beam_width=3, max_results=5)
        assert out.items == []
        assert out.scored == []

    def test_paths_use_labels_not_ids(self):
        leaf = _node("cluster-1", "deployment errors", _normalize([1.0, 0.0]),
                     items=[_item(0, _normalize([1.0, 0.0]))])
        root = _root_with([leaf])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=5)
        assert out.paths == [["deployment errors"]]
        # IDs must NOT appear in paths
        assert "cluster-1" not in [p for path in out.paths for p in path]


class TestPruning:
    def test_children_below_threshold_pruned(self):
        # Three top-level clusters: one matches, two don't.
        a = _node("a", "A", _normalize([1.0, 0.0]),
                  items=[_item(1, _normalize([1.0, 0.05]))])
        b = _node("b", "B", _normalize([0.0, 1.0]),
                  items=[_item(2, _normalize([0.0, 1.0]))])
        c = _node("c", "C", _normalize([-1.0, 0.0]),
                  items=[_item(3, _normalize([-1.0, 0.0]))])
        root = _root_with([a, b, c])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=5)
        # Only items from cluster A should be returned.
        assert {it.id for it in out.items} == {"item-1"}

    def test_beam_width_limits_branches(self):
        # Five clusters, all somewhat aligned with the query.
        children = [
            _node(f"c{i}", f"L{i}", _normalize([1.0, i * 0.05]),
                  items=[_item(i, _normalize([1.0, i * 0.05]))])
            for i in range(5)
        ]
        root = _root_with(children)
        cfg = MindmapConfig(similarity_threshold=0.5)
        # beam_width=2 keeps only the top-2 branches.
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=2, max_results=10)
        assert len(out.paths) == 2

    def test_no_root_children_pass_falls_back_to_root(self):
        # Two children both orthogonal to query.
        a = _node("a", "A", _normalize([0.0, 1.0]),
                  items=[_item(1, _normalize([0.0, 1.0]))])
        root = _root_with([a])
        cfg = MindmapConfig(similarity_threshold=0.9)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=5)
        # paths should contain at least the empty path (root)
        assert any(p == [] for p in out.paths)


class TestFiltering:
    def test_exact_match_filter(self):
        a = _node("a", "A", _normalize([1.0, 0.0]), items=[
            _item(1, _normalize([1.0, 0.0]), source="summary"),
            _item(2, _normalize([1.0, 0.0]), source="raw"),
            _item(3, _normalize([1.0, 0.0]), source="summary"),
        ])
        root = _root_with([a])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=10,
                          filter={"source": "summary"})
        assert {it.id for it in out.items} == {"item-1", "item-3"}

    def test_min_score_filter(self):
        a = _node("a", "A", _normalize([1.0, 0.0]), items=[
            _item(1, _normalize([1.0, 0.0])),     # sim=1.0
            _item(2, _normalize([1.0, 0.6])),     # sim≈0.857
            _item(3, _normalize([1.0, 1.5])),     # sim≈0.554
        ])
        root = _root_with([a])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=10,
                          min_score=0.8)
        ids = {it.id for it in out.items}
        assert "item-1" in ids
        assert "item-2" in ids
        assert "item-3" not in ids

    def test_max_results_slices(self):
        a = _node("a", "A", _normalize([1.0, 0.0]), items=[
            _item(i, _normalize([1.0, i * 0.01])) for i in range(10)
        ])
        root = _root_with([a])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=3)
        assert len(out.items) == 3


class TestDedup:
    def test_items_deduped_by_id(self):
        # Same item id in two clusters (degenerate, but tree shouldn't double-count).
        dup = _item(7, _normalize([1.0, 0.0]))
        a = _node("a", "A", _normalize([1.0, 0.0]), items=[dup])
        b = _node("b", "B", _normalize([1.0, 0.0]), items=[dup])
        root = _root_with([a, b])
        cfg = MindmapConfig(similarity_threshold=0.5)
        out = beam_search(root, _normalize([1.0, 0.0]), cfg, beam_width=3, max_results=10)
        assert [it.id for it in out.items] == ["item-7"]
