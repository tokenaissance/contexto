"""Extract Q:/A:/T: episode text from a Hermes WebhookPayload.

Hermes' `build_episode_payload` writes `data: {"messages": [...]}` — a flat list of
role-tagged messages. TS expects `data.userMessage` / `assistantMessages` / `toolMessages`,
a shape OpenClaw produces but Hermes does not. This extractor reads the flat shape
and emits the same Q:/A:/T: text TS produces.

Spec: §5 (extractor).
"""

from __future__ import annotations

from typing import Any

from ..helpers import normalize_message_text, strip_metadata_envelope
from ..types import WebhookPayload


def extract_episode_text(payload: WebhookPayload) -> str:
    """Return Q:/A:/T:-prefixed text or `""` for non-episode events."""
    event = payload.get("event") if isinstance(payload, dict) else None
    if not isinstance(event, dict):
        return ""
    if event.get("type") != "episode" or event.get("action") != "combined":
        return ""

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return ""
    messages = data.get("messages")
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = normalize_message_text(msg)
        if role == "user":
            if not text:
                continue
            stripped = strip_metadata_envelope(text)
            parts.append(f"Q: {stripped}")
        elif role == "assistant":
            if not text:
                continue
            parts.append(f"A: {text}")
        elif role == "tool":
            if not text:
                continue
            parts.append(f"T: {text}")
        # Other roles (system, etc.) ignored.

    return "\n".join(parts)


__all__ = ["extract_episode_text"]
