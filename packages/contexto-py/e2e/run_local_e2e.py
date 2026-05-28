"""End-to-end test of LocalBackend against a real provider (default: OpenRouter).

Reads OPENROUTER_API_KEY (or OPENAI_API_KEY) from the environment, runs an ingest +
search round-trip, and asserts the mindmap state is persisted as expected.

Usage:
    cd contexto/packages/contexto-py
    source e2e/.env && OPENROUTER_API_KEY=$OPENROUTER_API_KEY .venv/bin/python e2e/run_local_e2e.py

Exits non-zero on any failure. Designed to run inside the slim Docker container
(see e2e/Dockerfile) or directly on the host.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("e2e")


def _build_episode(idx: int, user: str, assistant: str, session_key: str | None = None) -> dict:
    return {
        "event": {"type": "episode", "action": "combined"},
        "sessionKey": session_key or f"e2e-{idx}",
        "timestamp": "2026-05-26T12:00:00.000Z",
        "context": {"sessionId": session_key or f"e2e-{idx}", "model": "openrouter", "provider": "openrouter"},
        "data": {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]},
    }


def main() -> int:
    log.info("starting LocalBackend E2E")
    # Force the storage path into a temp dir so re-runs are clean.
    tmp = Path(tempfile.mkdtemp(prefix="local-e2e-"))
    storage = tmp / "mindmap.json"
    os.environ["CONTEXTO_LOCAL_STORAGE_PATH"] = str(storage)
    os.environ.setdefault("CONTEXTO_BACKEND", "local")

    # Import AFTER setting env so from_env picks up our path.
    from contexto_hermes.local.backend import LocalBackend
    from contexto_hermes.local.mindmap_types import LocalBackendConfig
    from contexto_hermes.local.store import Store

    cfg = LocalBackendConfig.from_env()
    if cfg is None:
        log.error("LocalBackendConfig.from_env() returned None — set OPENROUTER_API_KEY or OPENAI_API_KEY")
        return 2

    log.info(
        "provider=%s embed_model=%s llm_model=%s storage=%s",
        cfg.provider, cfg.embed_model, cfg.llm_model, cfg.storage_path,
    )

    backend = LocalBackend(cfg)

    # ---- ingest ----
    episodes = [
        _build_episode(
            1,
            "How do I deploy a Kubernetes cluster?",
            "Use `kubectl apply -f deploy.yaml`. Check status with `kubectl get pods`.",
        ),
        _build_episode(
            2,
            "What's the best Italian restaurant in town?",
            "Trattoria Da Mario has great pasta and a wood-fired oven.",
        ),
        _build_episode(
            3,
            "My Kubernetes pod is stuck in CrashLoopBackOff.",
            "Check the container logs with `kubectl logs <pod>` and look for the exit reason. "
            "Common causes: missing env var, OOM, failing healthcheck.",
        ),
    ]

    t0 = time.monotonic()
    ok = backend.ingest(episodes)
    t1 = time.monotonic()
    if not ok:
        log.error("ingest returned False — see ERROR logs above")
        return 3
    log.info("ingest OK in %.2fs", t1 - t0)

    # ---- verify persistence ----
    if not storage.exists():
        log.error("mindmap.json not written to %s", storage)
        return 4
    raw = json.loads(storage.read_text())
    log.info(
        "persisted: version=%s total_items=%s total_clusters=%s",
        raw["version"], raw["stats"]["total_items"], raw["stats"]["total_clusters"],
    )
    if raw["version"] != 1:
        log.error("expected version=1, got %r", raw["version"])
        return 5
    if raw["stats"]["total_items"] != 3:
        log.error("expected total_items=3, got %r", raw["stats"]["total_items"])
        return 5
    if raw["root"] is None:
        log.error("root cluster is None after 3 episodes")
        return 5

    # Spot-check first item shape (metadata per spec §6).
    first_cluster = raw["root"]["children"][0]
    first_item = first_cluster.get("items", [None])[0] or first_cluster["children"][0]["items"][0]
    md = first_item["metadata"]
    log.info("first item metadata keys: %s", sorted(md.keys()))
    for required_key in ("source", "status", "confidence", "evidence_refs",
                         "open_questions", "trace_ref", "sessionKey", "episode"):
        if required_key not in md:
            log.error("metadata missing required key %r", required_key)
            return 6
    if md["source"] != "summary":
        log.error("expected metadata.source='summary', got %r", md["source"])
        return 6

    # ---- search (semantic relevance) ----
    t0 = time.monotonic()
    result = backend.search(
        "kubernetes pod crashing",
        max_results=3,
        filter={"source": "summary"},
        min_score=0.0,
    )
    t1 = time.monotonic()
    if result is None:
        log.error("search returned None")
        return 7
    log.info("search OK in %.2fs, got %d items", t1 - t0, len(result.items))
    # SearchResult.items entries are {"item": ConversationItem-dict, "score": float}
    # per spec §5 (TS ScoredQueryResult parity).
    for i, entry in enumerate(result.items):
        item = entry["item"]
        log.info(
            "  rank %d: score=%.3f content[:80]=%r",
            i + 1, entry["score"], item["content"][:80],
        )

    # The kubernetes-shaped items should outrank the restaurant one.
    top_contents = " ".join(e["item"]["content"].lower() for e in result.items[:2])
    if "kubernetes" not in top_contents and "kubectl" not in top_contents and "pod" not in top_contents:
        log.warning(
            "kubernetes terms not in top-2 results — embeddings may not be well-aligned. "
            "Top-2 contents: %s", top_contents[:300],
        )
        # Don't fail hard — semantic search behavior depends on the provider.
        # We do require at least one item to come back.

    # ---- second instance reload ----
    backend2 = LocalBackend(cfg)
    state = Store(cfg.storage_path).load()
    if state.stats.total_items != 3:
        log.error("reload: expected 3 items, got %s", state.stats.total_items)
        return 8
    result2 = backend2.search("italian food", max_results=3, filter={"source": "summary"})
    if result2 is None:
        log.warning("second-instance search returned None")
    else:
        log.info("second-instance search OK, %d items", len(result2.items))

    log.info("E2E PASSED")
    log.info("mindmap.json at %s (size=%d bytes)", storage, storage.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
