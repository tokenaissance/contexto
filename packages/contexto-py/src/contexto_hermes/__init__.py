"""contexto-hermes — Contexto context engine plugin for hermes-agent.

Plugin entry point. Hermes' `_EngineCollector` exec's this module and calls
`register(ctx)`; we wire a `ContextoEngine` instance into `ctx`.

The plugin supports two backends:
  - `remote` (default): HTTP to api.getcontexto.com. Requires CONTEXTO_API_KEY.
  - `local`: pure-Python pipeline with on-disk mindmap; requires an OpenAI or
    OpenRouter key. See `local/backend.py`.

The active backend is selected via CONTEXTO_BACKEND.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .engine import ContextoEngine

__all__ = ["ContextoEngine", "register", "__compatible_contexto_api__"]
__version__ = "0.2.0"

# Pinned `api.getcontexto.com` schema version compatible with this release.
# Bumped independently from `@ekai/contexto`'s semver.
__compatible_contexto_api__ = "2026-05"

logger = logging.getLogger("plugins.context_engine.contexto")


_VALID_BACKENDS = ("remote", "local")


def _resolve_backend() -> str:
    raw = os.environ.get("CONTEXTO_BACKEND", "").strip().lower()
    if not raw:
        return "remote"
    if raw not in _VALID_BACKENDS:
        logger.warning(
            "Invalid CONTEXTO_BACKEND=%r; falling back to 'remote'. "
            "Valid values: %s.",
            raw, ", ".join(_VALID_BACKENDS),
        )
        return "remote"
    return raw


def register(ctx: Any) -> None:
    """Plugin registration. Called by hermes-agent's context-engine loader."""
    backend = _resolve_backend()

    try:
        if backend == "local":
            engine = ContextoEngine.from_env_local()
        else:
            engine = ContextoEngine.from_env()
    except Exception as exc:
        logger.error(
            "Contexto plugin not registered: %s backend construction raised: %s",
            backend, exc, exc_info=True,
        )
        return

    if engine is None:
        if backend == "local":
            logger.error(
                "Contexto plugin (local) not registered: local config invalid. "
                "See prior log lines for the specific reason (provider key missing, "
                "unknown CONTEXTO_LOCAL_PROVIDER, etc.). "
                "Hermes will fall back to the default 'compressor' engine."
            )
        else:
            logger.error(
                "Contexto plugin not registered: CONTEXTO_API_KEY is not set. "
                "Hermes will fall back to the default 'compressor' engine. "
                "Get a key at https://getcontexto.com and `export CONTEXTO_API_KEY=...`."
            )
        return

    ctx.register_context_engine(engine)
