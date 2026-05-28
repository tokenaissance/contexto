"""Sanity test for plugin.yaml — required env_vars and name."""

from __future__ import annotations

from pathlib import Path

import yaml

PLUGIN_YAML = Path(__file__).parent.parent / "src" / "contexto_hermes" / "plugin.yaml"


def _load() -> dict:
    return yaml.safe_load(PLUGIN_YAML.read_text())


def test_name_is_contexto() -> None:
    assert _load()["name"] == "contexto"


def test_required_env_vars_present() -> None:
    data = _load()
    names = {ev["name"] for ev in data["env_vars"]}
    for required in (
        "CONTEXTO_API_KEY",
        "CONTEXTO_ENABLED",
        "CONTEXTO_MAX_CONTEXT_CHARS",
        "CONTEXTO_MIN_SCORE",
        "CONTEXTO_MAX_RESULTS",
        "CONTEXTO_SEARCH_TIMEOUT",
        "CONTEXTO_INGEST_TIMEOUT",
    ):
        assert required in names, f"missing env var: {required}"


def test_api_key_not_unconditionally_required() -> None:
    # As of v0.2.0, CONTEXTO_API_KEY is only needed when CONTEXTO_BACKEND=remote;
    # the local backend uses provider keys. So the manifest must NOT mark it required.
    data = _load()
    api_key_entry = next(ev for ev in data["env_vars"] if ev["name"] == "CONTEXTO_API_KEY")
    assert api_key_entry.get("required") is False


def test_backend_selector_declared() -> None:
    data = _load()
    names = {ev["name"] for ev in data["env_vars"]}
    assert "CONTEXTO_BACKEND" in names


def test_local_backend_env_vars_declared() -> None:
    data = _load()
    names = {ev["name"] for ev in data["env_vars"]}
    for required in (
        "CONTEXTO_LOCAL_PROVIDER",
        "CONTEXTO_LOCAL_STORAGE_PATH",
        "CONTEXTO_LOCAL_EMBED_MODEL",
        "CONTEXTO_LOCAL_LLM_MODEL",
        "CONTEXTO_LOCAL_SUMMARIZE",
        "CONTEXTO_LOCAL_SIMILARITY_THRESHOLD",
        "CONTEXTO_LOCAL_MAX_DEPTH",
        "CONTEXTO_LOCAL_MAX_CHILDREN",
        "CONTEXTO_LOCAL_REBUILD_INTERVAL",
        "CONTEXTO_LOCAL_BEAM_WIDTH",
        "CONTEXTO_LOCAL_EMBED_TIMEOUT",
        "CONTEXTO_LOCAL_LLM_TIMEOUT",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        assert required in names, f"missing env var: {required}"


def test_version_matches_package() -> None:
    import contexto_hermes
    assert _load()["version"] == contexto_hermes.__version__
