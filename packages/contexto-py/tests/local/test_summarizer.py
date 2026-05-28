"""Tests for the LLM summarizer."""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from contexto_hermes.local.summarizer import (
    SUMMARIZE_SYSTEM_PROMPT,
    Summarizer,
    build_synthetic_summary,
)


def _mock(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _success(content: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})


class TestRequest:
    def test_url_headers_method(self, base_config):
        captured: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return _success({"summary": "s", "status": "complete", "key_findings": ["x"], "confidence": 0.9})

        s = Summarizer(base_config, transport=_mock(handler))
        s.summarize("episode text")
        req = captured[0]
        assert str(req.url) == "https://api.openai.com/v1/chat/completions"
        assert req.method == "POST"
        assert req.headers["authorization"] == "Bearer sk-test"

    def test_body_has_temperature_and_response_format(self, base_config):
        captured: list[bytes] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req.content)
            return _success({"summary": "s", "status": "complete", "key_findings": ["x"], "confidence": 0.5})

        s = Summarizer(base_config, transport=_mock(handler))
        s.summarize("episode text")
        body = json.loads(captured[0])
        assert body["temperature"] == 0.2
        assert body["response_format"] == {"type": "json_object"}
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"][0] == {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT}
        assert body["messages"][1] == {"role": "user", "content": "episode text"}

    def test_system_prompt_verbatim(self):
        # Spot-check key invariants from the TS prompt
        assert "concise summarizer" in SUMMARIZE_SYSTEM_PROMPT
        assert '"status": "complete" | "partial" | "blocked"' in SUMMARIZE_SYSTEM_PROMPT
        assert "Respond ONLY with valid JSON, no markdown fences, no extra text." in SUMMARIZE_SYSTEM_PROMPT


class TestResponseParsing:
    def test_well_formed_summary(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return _success({
                "summary": "the assistant explained X",
                "key_findings": ["finding-a", "finding-b"],
                "status": "complete",
                "confidence": 0.91,
                "evidence_refs": [{"type": "file_ref", "value": "foo.py"}],
                "open_questions": ["q1"],
            })

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("text")
        assert out.summary == "the assistant explained X"
        assert out.key_findings == ["finding-a", "finding-b"]
        assert out.status == "complete"
        assert out.confidence == 0.91
        assert out.evidence_refs[0].type == "file_ref"
        assert out.evidence_refs[0].value == "foo.py"
        assert out.open_questions == ["q1"]

    def test_invalid_status_coerced_to_partial(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return _success({"summary": "s", "status": "weird", "key_findings": ["x"], "confidence": 0.5})

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("text")
        assert out.status == "partial"

    def test_invalid_confidence_coerced_to_half(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return _success({"summary": "s", "status": "complete", "key_findings": ["x"], "confidence": 1.5})

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("text")
        assert out.confidence == 0.5

    def test_evidence_refs_filtered_to_valid_dicts(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return _success({
                "summary": "s",
                "status": "complete",
                "key_findings": ["x"],
                "confidence": 0.5,
                "evidence_refs": [
                    {"type": "tool_ref", "value": "ok"},
                    {"type": 123, "value": "bad"},  # bad
                    "not a dict",
                ],
            })

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("text")
        assert len(out.evidence_refs) == 1
        assert out.evidence_refs[0].type == "tool_ref"


class TestFallback:
    def test_http_error_returns_fallback(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server boom")

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("text")
        assert "fallback" in out.key_findings[0]
        assert out.confidence == 0.0
        assert out.status == "partial"

    def test_parse_error_returns_fallback(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("hello")
        assert "fallback" in out.key_findings[0]

    def test_network_error_returns_fallback(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        s = Summarizer(base_config, transport=_mock(handler))
        out = s.summarize("x")
        assert "fallback" in out.key_findings[0]

    def test_fallback_truncates_long_text(self, base_config):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        s = Summarizer(base_config, transport=_mock(handler))
        long_text = "x" * 500
        out = s.summarize(long_text)
        assert out.summary.endswith("...")
        assert len(out.summary) == 203


class TestSyntheticSummary:
    def test_distinct_marker(self):
        out = build_synthetic_summary("hello")
        assert out.summary == "hello"
        assert out.key_findings == ["Episode processed (summarization disabled)"]
        assert out.status == "partial"
        assert out.confidence == 0.0
        assert out.evidence_refs == []

    def test_truncates_long_text(self):
        out = build_synthetic_summary("x" * 500)
        assert out.summary.endswith("...")
        assert len(out.summary) == 203
