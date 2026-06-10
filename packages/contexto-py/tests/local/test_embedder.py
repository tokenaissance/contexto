"""Tests for the embeddings client."""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from contexto_hermes.local.embedder import Embedder, EmbedError


def _mock(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestRequest:
    def test_openai_url_and_headers(self, base_config):
        captured: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

        emb = Embedder(base_config, transport=_mock(handler))
        emb.embed("hello")
        req = captured[0]
        assert str(req.url) == "https://api.openai.com/v1/embeddings"
        assert req.method == "POST"
        assert req.headers["authorization"] == "Bearer sk-test"
        assert req.headers["content-type"].startswith("application/json")

    def test_openrouter_url(self, openrouter_config):
        captured: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

        emb = Embedder(openrouter_config, transport=_mock(handler))
        emb.embed("hi")
        assert str(captured[0].url) == "https://openrouter.ai/api/v1/embeddings"
        assert captured[0].headers["authorization"] == "Bearer sk-or-test"

    def test_body_uses_model_and_input(self, base_config):
        captured: list[bytes] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req.content)
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

        emb = Embedder(base_config, transport=_mock(handler))
        emb.embed("hello world")
        body = json.loads(captured[0])
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == "hello world"

    def test_openrouter_uses_namespaced_model(self, openrouter_config):
        captured: list[bytes] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req.content)
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

        emb = Embedder(openrouter_config, transport=_mock(handler))
        emb.embed("hi")
        body = json.loads(captured[0])
        assert body["model"] == "openai/text-embedding-3-small"


class TestResponse:
    def test_parses_data_zero_embedding(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [1.5, -0.5, 0.0]}]})

        emb = Embedder(base_config, transport=_mock(handler))
        assert emb.embed("x") == [1.5, -0.5, 0.0]

    def test_http_error_raises_EmbedError(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limit")

        emb = Embedder(base_config, transport=_mock(handler))
        with pytest.raises(EmbedError):
            emb.embed("x")

    def test_network_error_raises_EmbedError(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route")

        emb = Embedder(base_config, transport=_mock(handler))
        with pytest.raises(EmbedError):
            emb.embed("x")

    def test_malformed_response_raises(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})  # empty list

        emb = Embedder(base_config, transport=_mock(handler))
        with pytest.raises(EmbedError):
            emb.embed("x")

    def test_non_json_response_raises(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        emb = Embedder(base_config, transport=_mock(handler))
        with pytest.raises(EmbedError):
            emb.embed("x")
