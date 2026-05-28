"""Tests for backend-aware register()."""

from __future__ import annotations

import importlib
import logging

import pytest


class _CapturingCtx:
    def __init__(self) -> None:
        self.registered: list = []

    def register_context_engine(self, engine) -> None:
        self.registered.append(engine)


_LOCAL_ENV_VARS = (
    "CONTEXTO_BACKEND",
    "CONTEXTO_API_KEY",
    "CONTEXTO_LOCAL_PROVIDER",
    "CONTEXTO_LOCAL_STORAGE_PATH",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "HERMES_HOME",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for var in _LOCAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Always sandbox the local storage path so tests don't write to ~/.hermes.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _reimport():
    import contexto_hermes
    return importlib.reload(contexto_hermes)


class TestBackendSelection:
    def test_default_is_remote(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_API_KEY", "ckai_abc")
        ctx = _CapturingCtx()
        _reimport().register(ctx)
        assert len(ctx.registered) == 1
        assert ctx.registered[0].name == "contexto"
        # Remote backend is the default.
        from contexto_hermes.client import RemoteBackend
        assert isinstance(ctx.registered[0].client, RemoteBackend)

    def test_local_with_openai_key_registers_local_backend(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_BACKEND", "local")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ctx = _CapturingCtx()
        _reimport().register(ctx)
        assert len(ctx.registered) == 1
        from contexto_hermes.local.backend import LocalBackend
        assert isinstance(ctx.registered[0].client, LocalBackend)

    def test_local_with_openrouter_key(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_BACKEND", "local")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        ctx = _CapturingCtx()
        _reimport().register(ctx)
        assert len(ctx.registered) == 1
        from contexto_hermes.local.backend import LocalBackend
        assert isinstance(ctx.registered[0].client, LocalBackend)
        assert ctx.registered[0].client._config.provider == "openrouter"

    def test_local_without_any_key_does_not_register(self, monkeypatch, caplog):
        monkeypatch.setenv("CONTEXTO_BACKEND", "local")
        ctx = _CapturingCtx()
        with caplog.at_level(logging.ERROR, logger="plugins.context_engine.contexto"):
            _reimport().register(ctx)
        assert ctx.registered == []
        assert any("local" in r.message.lower() for r in caplog.records)


class TestBackendValidation:
    def test_invalid_backend_falls_back_to_remote(self, monkeypatch, caplog):
        monkeypatch.setenv("CONTEXTO_BACKEND", "redis")
        monkeypatch.setenv("CONTEXTO_API_KEY", "ckai_abc")
        ctx = _CapturingCtx()
        with caplog.at_level(logging.WARNING, logger="plugins.context_engine.contexto"):
            _reimport().register(ctx)
        assert len(ctx.registered) == 1
        from contexto_hermes.client import RemoteBackend
        assert isinstance(ctx.registered[0].client, RemoteBackend)
        assert any("CONTEXTO_BACKEND" in r.message for r in caplog.records)

    def test_remote_explicit_works(self, monkeypatch):
        monkeypatch.setenv("CONTEXTO_BACKEND", "remote")
        monkeypatch.setenv("CONTEXTO_API_KEY", "ckai_abc")
        ctx = _CapturingCtx()
        _reimport().register(ctx)
        assert len(ctx.registered) == 1

    def test_remote_without_api_key_does_not_register(self, monkeypatch, caplog):
        monkeypatch.setenv("CONTEXTO_BACKEND", "remote")
        ctx = _CapturingCtx()
        with caplog.at_level(logging.ERROR, logger="plugins.context_engine.contexto"):
            _reimport().register(ctx)
        assert ctx.registered == []
        assert any("CONTEXTO_API_KEY" in r.message for r in caplog.records)
