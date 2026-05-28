"""Shared fixtures for local-backend tests."""

from __future__ import annotations

from typing import Any

import pytest

from contexto_hermes.local.mindmap_types import LocalBackendConfig, MindmapConfig


@pytest.fixture
def base_config(tmp_path) -> LocalBackendConfig:
    """A LocalBackendConfig wired for tests — no real HTTP calls."""
    return LocalBackendConfig(
        storage_path=str(tmp_path / "mindmap.json"),
        provider="openai",
        api_key="sk-test",
        embed_base_url="https://api.openai.com/v1",
        llm_base_url="https://api.openai.com/v1",
        embed_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        summarize=True,
        mindmap=MindmapConfig(),
        beam_width=3,
        embed_timeout=5.0,
        llm_timeout=10.0,
    )


@pytest.fixture
def openrouter_config(tmp_path) -> LocalBackendConfig:
    return LocalBackendConfig(
        storage_path=str(tmp_path / "mindmap.json"),
        provider="openrouter",
        api_key="sk-or-test",
        embed_base_url="https://openrouter.ai/api/v1",
        llm_base_url="https://openrouter.ai/api/v1",
        embed_model="openai/text-embedding-3-small",
        llm_model="openai/gpt-4o-mini",
        summarize=True,
        mindmap=MindmapConfig(),
        beam_width=3,
        embed_timeout=5.0,
        llm_timeout=10.0,
    )
