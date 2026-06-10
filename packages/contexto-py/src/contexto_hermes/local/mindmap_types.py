"""Dataclasses for the local mindmap backend.

Module named `mindmap_types.py` to avoid colliding with `contexto_hermes.types`.
The on-disk JSON schema (spec §8) is the source of truth for field semantics;
these dataclasses mirror it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import _env_bool, _env_float, _env_int

logger = logging.getLogger("plugins.context_engine.contexto")


# Provider-specific defaults. Spec §5.
_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "embed_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "embed_base_url": "https://api.openai.com/v1",
        "llm_base_url": "https://api.openai.com/v1",
    },
    "openrouter": {
        "embed_model": "openai/text-embedding-3-small",
        "llm_model": "openai/gpt-4o-mini",
        "embed_base_url": "https://openrouter.ai/api/v1",
        "llm_base_url": "https://openrouter.ai/api/v1",
    },
}


def _default_storage_path() -> str:
    """Resolve `$HERMES_HOME/data/contexto/mindmap.json`, falling back to `~/.hermes`."""
    base = os.environ.get("HERMES_HOME") or "~/.hermes"
    return str(Path(base).expanduser() / "data" / "contexto" / "mindmap.json")


@dataclass
class MindmapConfig:
    """Mirrors TS `DEFAULT_CONFIG` in packages/mindmap/src/types.ts:80-85."""

    similarity_threshold: float = 0.65
    max_depth: int = 4
    max_children: int = 10  # informational; not enforced (matches TS)
    rebuild_interval: int = 50


@dataclass
class LocalBackendConfig:
    """Configuration for the local mindmap backend.

    `from_env()` returns None when credentials/config are unusable. Provider/key
    resolution rules are in spec §7.

    `embed_model` and `llm_model` are nullable: `None` means "use the provider
    default" (resolved at call time by `Embedder`/`Summarizer`). `from_env`
    leaves them as None when the user did not override; direct construction may
    do the same.
    """

    storage_path: str
    provider: str  # "openai" | "openrouter"
    api_key: str
    embed_base_url: str
    llm_base_url: str
    embed_model: str | None = None
    llm_model: str | None = None
    summarize: bool = True
    mindmap: MindmapConfig = field(default_factory=MindmapConfig)
    beam_width: int = 3
    embed_timeout: float = 30.0
    llm_timeout: float = 60.0

    def resolved_embed_model(self) -> str:
        """Provider default when `embed_model` is None."""
        if self.embed_model:
            return self.embed_model
        return _PROVIDER_DEFAULTS[self.provider]["embed_model"]

    def resolved_llm_model(self) -> str:
        """Provider default when `llm_model` is None."""
        if self.llm_model:
            return self.llm_model
        return _PROVIDER_DEFAULTS[self.provider]["llm_model"]

    @classmethod
    def from_env(cls) -> "LocalBackendConfig | None":
        """Read CONTEXTO_LOCAL_* env vars. Returns None on unusable config."""
        # Provider selection (spec §7 table)
        explicit_provider = os.environ.get("CONTEXTO_LOCAL_PROVIDER", "").strip().lower()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

        provider: str
        api_key: str
        if explicit_provider:
            if explicit_provider == "openai":
                if not openai_key:
                    logger.error(
                        "CONTEXTO_LOCAL_PROVIDER=openai but OPENAI_API_KEY is unset."
                    )
                    return None
                provider, api_key = "openai", openai_key
            elif explicit_provider == "openrouter":
                if not openrouter_key:
                    logger.error(
                        "CONTEXTO_LOCAL_PROVIDER=openrouter but OPENROUTER_API_KEY is unset."
                    )
                    return None
                provider, api_key = "openrouter", openrouter_key
            else:
                logger.error(
                    "Unknown CONTEXTO_LOCAL_PROVIDER=%r (expected 'openai' or 'openrouter').",
                    explicit_provider,
                )
                return None
        else:
            # Implicit: prefer openrouter, then openai.
            if openrouter_key:
                provider, api_key = "openrouter", openrouter_key
            elif openai_key:
                provider, api_key = "openai", openai_key
            else:
                logger.error(
                    "Local backend requires OPENROUTER_API_KEY or OPENAI_API_KEY."
                )
                return None

        defaults = _PROVIDER_DEFAULTS[provider]
        # None ⇒ use provider default at call time (resolved_embed_model / resolved_llm_model).
        embed_model = os.environ.get("CONTEXTO_LOCAL_EMBED_MODEL", "").strip() or None
        llm_model = os.environ.get("CONTEXTO_LOCAL_LLM_MODEL", "").strip() or None
        storage_path = os.environ.get("CONTEXTO_LOCAL_STORAGE_PATH", "").strip() or _default_storage_path()

        mindmap = MindmapConfig(
            similarity_threshold=_env_float(
                "CONTEXTO_LOCAL_SIMILARITY_THRESHOLD",
                default=0.65,
                minimum=0.0,
                maximum=1.0,
            ),
            max_depth=_env_int("CONTEXTO_LOCAL_MAX_DEPTH", default=4, minimum=1),
            max_children=_env_int("CONTEXTO_LOCAL_MAX_CHILDREN", default=10, minimum=1),
            rebuild_interval=_env_int("CONTEXTO_LOCAL_REBUILD_INTERVAL", default=50, minimum=1),
        )

        return cls(
            storage_path=storage_path,
            provider=provider,
            api_key=api_key,
            embed_model=embed_model,
            llm_model=llm_model,
            embed_base_url=defaults["embed_base_url"],
            llm_base_url=defaults["llm_base_url"],
            summarize=_env_bool("CONTEXTO_LOCAL_SUMMARIZE", default=True),
            mindmap=mindmap,
            beam_width=_env_int("CONTEXTO_LOCAL_BEAM_WIDTH", default=3, minimum=1),
            embed_timeout=_env_float("CONTEXTO_LOCAL_EMBED_TIMEOUT", default=30.0, minimum=0.0),
            llm_timeout=_env_float("CONTEXTO_LOCAL_LLM_TIMEOUT", default=60.0, minimum=0.0),
        )


@dataclass
class EvidenceRef:
    type: str  # "episode_ref" | "tool_ref" | "file_ref" | "trace_ref"
    value: str


@dataclass
class EpisodeSummary:
    """Mirrors TS EpisodeSummary in packages/contexto/src/local/types.ts."""

    summary: str
    key_findings: list[str]
    status: str  # "complete" | "partial" | "blocked"
    confidence: float
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    open_questions: list[str] | None = None


@dataclass
class ConversationItem:
    id: str
    role: str
    content: str
    embedding: list[float]
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterNode:
    id: str
    label: str
    centroid: list[float]
    children: list["ClusterNode"] = field(default_factory=list)
    items: list[ConversationItem] = field(default_factory=list)
    depth: int = 0
    item_count: int = 0


@dataclass
class StoreStats:
    total_items: int = 0
    total_clusters: int = 0
    inserts_since_rebuild: int = 0


@dataclass
class StoreState:
    version: int = 1
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    root: ClusterNode | None = None
    stats: StoreStats = field(default_factory=StoreStats)
