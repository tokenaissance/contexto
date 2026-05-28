"""Local mindmap backend — runs embeddings + summarization client-side.

Pure-Python; numpy + scipy + httpx. No Contexto-hosted API call.
"""

from __future__ import annotations

from .mindmap_types import (
    ClusterNode,
    ConversationItem,
    EpisodeSummary,
    EvidenceRef,
    LocalBackendConfig,
    MindmapConfig,
    StoreState,
    StoreStats,
)

__all__ = [
    "ClusterNode",
    "ConversationItem",
    "EpisodeSummary",
    "EvidenceRef",
    "LocalBackendConfig",
    "MindmapConfig",
    "StoreState",
    "StoreStats",
]
