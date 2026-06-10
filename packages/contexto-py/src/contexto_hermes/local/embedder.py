"""Embeddings client. POST to {/embeddings} on OpenAI or OpenRouter.

Per-call `httpx.Client` (sync). Provider-specific base URLs + default models live
in `LocalBackendConfig`. The embedder raises on HTTP / parse failures; the
caller (LocalBackend) wraps these for the never-raises contract.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .mindmap_types import LocalBackendConfig

logger = logging.getLogger("plugins.context_engine.contexto")


class EmbedError(RuntimeError):
    """Raised when the embeddings endpoint fails (HTTP / parse / network)."""


class Embedder:
    def __init__(
        self,
        config: LocalBackendConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    def embed(self, text: str) -> list[float]:
        """Return a single vector. Raises EmbedError on any failure."""
        url = f"{self._config.embed_base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        body: dict[str, Any] = {
            "model": self._config.resolved_embed_model(),
            "input": text,
        }

        try:
            with self._client() as client:
                response = client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise EmbedError(f"embed network error: {exc}") from exc

        if not response.is_success:
            preview = ""
            try:
                preview = response.text[:200]
            except Exception:
                pass
            raise EmbedError(
                f"embed HTTP {response.status_code}: {preview}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise EmbedError(f"embed response not JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise EmbedError("embed response top-level was not an object")

        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise EmbedError("embed response missing data[]")
        first = items[0]
        if not isinstance(first, dict):
            raise EmbedError("embed response data[0] not an object")
        vec = first.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise EmbedError("embed response missing data[0].embedding")
        try:
            return [float(x) for x in vec]
        except (TypeError, ValueError) as exc:
            raise EmbedError(f"embed vector contained non-numeric: {exc}") from exc

    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": self._config.embed_timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)


__all__ = ["Embedder", "EmbedError"]
