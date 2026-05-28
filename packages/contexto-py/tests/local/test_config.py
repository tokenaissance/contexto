"""Tests for LocalBackendConfig.from_env — spec §7 resolution rules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from contexto_hermes.local.mindmap_types import LocalBackendConfig, MindmapConfig


_LOCAL_ENV_VARS = (
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
    "HERMES_HOME",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _LOCAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestProviderResolution:
    def test_implicit_prefers_openrouter_when_both_keys_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.provider == "openrouter"
        assert cfg.api_key == "sk-or"
        # No override → field stays None; provider default resolved at call time.
        assert cfg.embed_model is None
        assert cfg.llm_model is None
        assert cfg.resolved_embed_model() == "openai/text-embedding-3-small"
        assert cfg.resolved_llm_model() == "openai/gpt-4o-mini"

    def test_implicit_falls_back_to_openai_when_only_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.provider == "openai"
        assert cfg.embed_model is None
        assert cfg.llm_model is None
        assert cfg.resolved_embed_model() == "text-embedding-3-small"
        assert cfg.resolved_llm_model() == "gpt-4o-mini"

    def test_nullable_models_resolve_to_provider_defaults_directly(self):
        # Spec §11: `None` means "use provider default" — verify the contract
        # holds when constructing the dataclass directly (not via from_env).
        cfg = LocalBackendConfig(
            storage_path="/tmp/x.json",
            provider="openai",
            api_key="sk",
            embed_base_url="https://api.openai.com/v1",
            llm_base_url="https://api.openai.com/v1",
        )
        assert cfg.embed_model is None
        assert cfg.llm_model is None
        assert cfg.resolved_embed_model() == "text-embedding-3-small"
        assert cfg.resolved_llm_model() == "gpt-4o-mini"

    def test_implicit_returns_none_when_no_key(self):
        assert LocalBackendConfig.from_env() is None

    def test_explicit_openai_uses_openai_key(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_LOCAL_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")  # ignored
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-openai"

    def test_explicit_openrouter_uses_openrouter_key(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_LOCAL_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.provider == "openrouter"
        assert cfg.api_key == "sk-or"

    def test_explicit_openai_missing_key_returns_none(self, monkeypatch, caplog):
        monkeypatch.setenv("CONTEXTO_LOCAL_PROVIDER", "openai")
        # only openrouter present; must NOT silently fall back
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        cfg = LocalBackendConfig.from_env()
        assert cfg is None

    def test_explicit_openrouter_missing_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_LOCAL_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")  # MUST NOT fall back
        cfg = LocalBackendConfig.from_env()
        assert cfg is None

    def test_unknown_provider_returns_none(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_LOCAL_PROVIDER", "gemini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        cfg = LocalBackendConfig.from_env()
        assert cfg is None

    def test_provider_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_LOCAL_PROVIDER", "OpenAI")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.provider == "openai"


class TestStoragePathDefault:
    def test_default_uses_hermes_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.storage_path == str(tmp_path / "data" / "contexto" / "mindmap.json")

    def test_default_falls_back_to_home_hermes(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.storage_path == str(
            Path("~/.hermes").expanduser() / "data" / "contexto" / "mindmap.json"
        )

    def test_explicit_storage_path_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("CONTEXTO_LOCAL_STORAGE_PATH", "/tmp/explicit/path.json")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.storage_path == "/tmp/explicit/path.json"


class TestModelOverrides:
    def test_embed_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("CONTEXTO_LOCAL_EMBED_MODEL", "custom-embed")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.embed_model == "custom-embed"

    def test_llm_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("CONTEXTO_LOCAL_LLM_MODEL", "custom-llm")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.llm_model == "custom-llm"


class TestMindmapTunables:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.mindmap == MindmapConfig(
            similarity_threshold=0.65,
            max_depth=4,
            max_children=10,
            rebuild_interval=50,
        )
        assert cfg.beam_width == 3
        assert cfg.embed_timeout == 30.0
        assert cfg.llm_timeout == 60.0
        assert cfg.summarize is True

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("CONTEXTO_LOCAL_SIMILARITY_THRESHOLD", "0.75")
        monkeypatch.setenv("CONTEXTO_LOCAL_MAX_DEPTH", "5")
        monkeypatch.setenv("CONTEXTO_LOCAL_MAX_CHILDREN", "20")
        monkeypatch.setenv("CONTEXTO_LOCAL_REBUILD_INTERVAL", "100")
        monkeypatch.setenv("CONTEXTO_LOCAL_BEAM_WIDTH", "5")
        monkeypatch.setenv("CONTEXTO_LOCAL_EMBED_TIMEOUT", "15")
        monkeypatch.setenv("CONTEXTO_LOCAL_LLM_TIMEOUT", "120")
        monkeypatch.setenv("CONTEXTO_LOCAL_SUMMARIZE", "false")
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.mindmap.similarity_threshold == 0.75
        assert cfg.mindmap.max_depth == 5
        assert cfg.mindmap.max_children == 20
        assert cfg.mindmap.rebuild_interval == 100
        assert cfg.beam_width == 5
        assert cfg.embed_timeout == 15.0
        assert cfg.llm_timeout == 120.0
        assert cfg.summarize is False

    def test_invalid_threshold_uses_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("CONTEXTO_LOCAL_SIMILARITY_THRESHOLD", "2.0")  # out of range
        cfg = LocalBackendConfig.from_env()
        assert cfg is not None
        assert cfg.mindmap.similarity_threshold == 0.65
