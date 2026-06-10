"""Beam search retrieval over the cluster tree.

Port of TS `queryMindmapMultiBranch`. Returns a flat list of scored items plus
`paths` — a list of label paths (not IDs) leading to each terminal node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .clustering import _collect_items, _cosine_sim
from .mindmap_types import ClusterNode, ConversationItem, MindmapConfig


@dataclass
class ScoredItem:
    item: ConversationItem
    score: float


@dataclass
class BeamResult:
    items: list[ConversationItem]
    paths: list[list[str]] = field(default_factory=list)
    scored: list[ScoredItem] = field(default_factory=list)


@dataclass
class _BeamEntry:
    node: ClusterNode
    path: list[str]


def beam_search(
    root: ClusterNode,
    query_embedding: list[float],
    config: MindmapConfig,
    *,
    beam_width: int,
    max_results: int,
    filter: dict[str, Any] | None = None,
    min_score: float | None = None,
) -> BeamResult:
    """Run beam search; return scored items + label paths."""
    if root is None:
        return BeamResult(items=[], paths=[], scored=[])

    threshold = config.similarity_threshold
    terminals: list[_BeamEntry] = []

    # Seed beam with root's children that pass the threshold.
    root_candidates = sorted(
        (
            (child, _cosine_sim(query_embedding, child.centroid))
            for child in root.children
        ),
        key=lambda c: c[1],
        reverse=True,
    )
    qualified = [(c, s) for c, s in root_candidates if s >= threshold][:beam_width]

    if not qualified:
        # Fall back to collecting from root itself.
        terminals.append(_BeamEntry(node=root, path=[]))
        beam: list[_BeamEntry] = []
    else:
        beam = [_BeamEntry(node=c, path=[c.label]) for c, _ in qualified]

    # Expand level by level.
    while beam:
        next_candidates: list[tuple[_BeamEntry, float]] = []
        for entry in beam:
            if not entry.node.children:
                terminals.append(entry)
                continue
            child_scores = [
                (child, _cosine_sim(query_embedding, child.centroid))
                for child in entry.node.children
            ]
            qualified_children = [(c, s) for c, s in child_scores if s >= threshold]
            if not qualified_children:
                terminals.append(entry)
                continue
            for child, sim in qualified_children:
                next_candidates.append((
                    _BeamEntry(node=child, path=entry.path + [child.label]),
                    sim,
                ))
        next_candidates.sort(key=lambda pair: pair[1], reverse=True)
        beam = [entry for entry, _ in next_candidates[:beam_width]]

    # Gather items from all terminal nodes, dedup by id.
    seen: set[str] = set()
    all_items: list[ConversationItem] = []
    for terminal in terminals:
        for item in _collect_items(terminal.node):
            if item.id in seen:
                continue
            seen.add(item.id)
            all_items.append(item)

    # Apply metadata filter (exact match).
    if filter:
        all_items = [
            it for it in all_items
            if all(it.metadata.get(k) == v for k, v in filter.items())
        ]

    scored = [
        ScoredItem(item=it, score=_cosine_sim(query_embedding, it.embedding))
        for it in all_items
    ]
    scored.sort(key=lambda s: s.score, reverse=True)

    if min_score is not None:
        scored = [s for s in scored if s.score >= min_score]

    scored = scored[:max_results]
    paths = [t.path for t in terminals]
    return BeamResult(items=[s.item for s in scored], paths=paths, scored=scored)


__all__ = ["beam_search", "BeamResult", "ScoredItem"]
