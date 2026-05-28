"""LLM summarization client + synthetic-summary fallback.

Mirrors TS `summarizeEpisode` in packages/contexto/src/local/summarizer.ts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .mindmap_types import EpisodeSummary, EvidenceRef, LocalBackendConfig

logger = logging.getLogger("plugins.context_engine.contexto")


# Verbatim from TS summarizer.ts:16-32.
SUMMARIZE_SYSTEM_PROMPT = """You are a concise summarizer. Given a conversation episode (user question + assistant answer + tool outputs), produce a JSON object with exactly these fields:

{
  "status": "complete" | "partial" | "blocked",
  "summary": "<concise one-paragraph summary of what happened in this episode>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "evidence_refs": [{"type": "<episode_ref|tool_ref|file_ref|trace_ref>", "value": "<reference>"}],
  "open_questions": ["<optional unresolved question>"],
  "confidence": <0.0 to 1.0>
}

Rules:
- Set status to "complete" if the episode fully resolved the user's request, "partial" if only partly, "blocked" if unable to proceed.
- summary should be 1-3 sentences capturing the essence.
- key_findings should have at least one entry.
- evidence_refs should reference relevant tools, files, or episodes mentioned.
- Respond ONLY with valid JSON, no markdown fences, no extra text."""

_VALID_STATUSES = ("complete", "partial", "blocked")


class Summarizer:
    def __init__(
        self,
        config: LocalBackendConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    def summarize(self, text: str) -> EpisodeSummary:
        """Run an LLM summarization. Never raises — returns fallback on error."""
        url = f"{self._config.llm_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        body: dict[str, Any] = {
            "model": self._config.resolved_llm_model(),
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }

        try:
            with self._client() as client:
                response = client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            logger.warning("[contexto:local] summarize network error: %s", exc)
            return _build_fallback(text)
        except Exception as exc:  # never-raises contract bubbles up; be safe
            logger.warning("[contexto:local] summarize unexpected error: %s", exc)
            return _build_fallback(text)

        if not response.is_success:
            preview = ""
            try:
                preview = response.text[:200]
            except Exception:
                pass
            logger.warning(
                "[contexto:local] summarize HTTP %d: %s",
                response.status_code, preview,
            )
            return _build_fallback(text)

        try:
            envelope = response.json()
        except ValueError as exc:
            logger.warning("[contexto:local] summarize response not JSON: %s", exc)
            return _build_fallback(text)

        try:
            raw = envelope["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError):
            logger.warning("[contexto:local] summarize response missing choices/message/content")
            return _build_fallback(text)

        if not raw:
            logger.warning("[contexto:local] summarize empty content")
            return _build_fallback(text)

        return _parse_summary(raw, text)

    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": self._config.llm_timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)


def _parse_summary(raw: str, original: str) -> EpisodeSummary:
    """Parse the LLM JSON; graceful degradation per TS parseSummary."""
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        logger.warning("[contexto:local] failed to parse summary JSON: %s", exc)
        return _build_fallback(original)
    if not isinstance(parsed, dict):
        return _build_fallback(original)

    summary_text = parsed.get("summary")
    if not isinstance(summary_text, str) or not summary_text:
        summary_text = original[:200]

    raw_findings = parsed.get("key_findings")
    if isinstance(raw_findings, list) and raw_findings:
        key_findings = [str(f) for f in raw_findings]
    else:
        key_findings = ["Episode processed"]

    raw_status = parsed.get("status")
    status = raw_status if raw_status in _VALID_STATUSES else "partial"

    raw_conf = parsed.get("confidence")
    if isinstance(raw_conf, (int, float)) and 0.0 <= raw_conf <= 1.0:
        confidence = float(raw_conf)
    else:
        confidence = 0.5

    evidence_refs: list[EvidenceRef] = []
    raw_refs = parsed.get("evidence_refs")
    if isinstance(raw_refs, list):
        for ref in raw_refs:
            if (
                isinstance(ref, dict)
                and isinstance(ref.get("type"), str)
                and isinstance(ref.get("value"), str)
            ):
                evidence_refs.append(EvidenceRef(type=ref["type"], value=ref["value"]))

    raw_questions = parsed.get("open_questions")
    open_questions: list[str] | None
    if isinstance(raw_questions, list):
        open_questions = [q for q in raw_questions if isinstance(q, str)]
    else:
        open_questions = None

    return EpisodeSummary(
        summary=summary_text,
        key_findings=key_findings,
        status=status,
        confidence=confidence,
        evidence_refs=evidence_refs,
        open_questions=open_questions,
    )


def _build_fallback(text: str) -> EpisodeSummary:
    """Fallback summary used on LLM failure. Matches TS buildFallback exactly."""
    truncated = text[:200] + ("..." if len(text) > 200 else "")
    return EpisodeSummary(
        summary=truncated,
        key_findings=["Episode processed (fallback — LLM summarization unavailable)"],
        status="partial",
        confidence=0.0,
        evidence_refs=[],
        open_questions=None,
    )


def build_synthetic_summary(text: str) -> EpisodeSummary:
    """Build a summary from raw text without any LLM call.

    Used when `CONTEXTO_LOCAL_SUMMARIZE=false`. The distinct `key_findings` marker
    distinguishes this from the LLM-failure fallback.
    """
    truncated = text[:200] + ("..." if len(text) > 200 else "")
    return EpisodeSummary(
        summary=truncated,
        key_findings=["Episode processed (summarization disabled)"],
        status="partial",
        confidence=0.0,
        evidence_refs=[],
        open_questions=None,
    )


__all__ = [
    "Summarizer",
    "SUMMARIZE_SYSTEM_PROMPT",
    "build_synthetic_summary",
]
