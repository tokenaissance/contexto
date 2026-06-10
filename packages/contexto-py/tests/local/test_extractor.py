"""Tests for extract_episode_text."""

from __future__ import annotations

from contexto_hermes.local.extractor import extract_episode_text


def _payload(messages, event_type="episode", action="combined"):
    return {
        "event": {"type": event_type, "action": action},
        "sessionKey": "s",
        "timestamp": "2026-05-23T00:00:00.000Z",
        "context": {"sessionId": "s"},
        "data": {"messages": messages},
    }


class TestEventGate:
    def test_non_episode_event_returns_empty(self):
        p = _payload([{"role": "user", "content": "hi"}], event_type="metric")
        assert extract_episode_text(p) == ""

    def test_non_combined_action_returns_empty(self):
        p = _payload([{"role": "user", "content": "hi"}], action="start")
        assert extract_episode_text(p) == ""

    def test_missing_event_returns_empty(self):
        assert extract_episode_text({"data": {"messages": []}}) == ""

    def test_missing_data_returns_empty(self):
        assert extract_episode_text({"event": {"type": "episode", "action": "combined"}}) == ""

    def test_non_list_messages_returns_empty(self):
        p = {
            "event": {"type": "episode", "action": "combined"},
            "data": {"messages": "not a list"},
        }
        assert extract_episode_text(p) == ""


class TestPrefixes:
    def test_q_prefix_on_user(self):
        out = extract_episode_text(_payload([{"role": "user", "content": "hello"}]))
        assert out == "Q: hello"

    def test_a_prefix_on_assistant(self):
        out = extract_episode_text(_payload([{"role": "assistant", "content": "hi"}]))
        assert out == "A: hi"

    def test_t_prefix_on_tool(self):
        out = extract_episode_text(_payload([{"role": "tool", "content": "result"}]))
        assert out == "T: result"

    def test_other_roles_ignored(self):
        out = extract_episode_text(_payload([
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "kept"},
        ]))
        assert out == "Q: kept"

    def test_q_a_t_ordering_preserved(self):
        out = extract_episode_text(_payload([
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Looking it up."},
            {"role": "tool", "content": "Sunny, 22C"},
            {"role": "assistant", "content": "It's sunny."},
        ]))
        assert out == (
            "Q: What's the weather?\n"
            "A: Looking it up.\n"
            "T: Sunny, 22C\n"
            "A: It's sunny."
        )


class TestContentShapes:
    def test_list_content_blocks_concatenated(self):
        out = extract_episode_text(_payload([
            {"role": "assistant", "content": [
                {"type": "text", "text": "part1"},
                {"type": "text", "text": "part2"},
            ]},
        ]))
        assert out == "A: part1part2"

    def test_list_content_skips_non_text_blocks(self):
        out = extract_episode_text(_payload([
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "1", "name": "foo"},  # skipped
            ]},
        ]))
        assert out == "A: hello"

    def test_none_content_skipped(self):
        out = extract_episode_text(_payload([
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "real"},
        ]))
        assert out == "A: real"

    def test_empty_user_content_skipped(self):
        out = extract_episode_text(_payload([
            {"role": "user", "content": ""},
            {"role": "user", "content": "real"},
        ]))
        assert out == "Q: real"


class TestEnvelopeStripping:
    def test_user_metadata_envelope_stripped(self):
        envelope = (
            "Sender (untrusted metadata): ```json\n{\"name\": \"x\"}\n```\n"
            "Actual user question?"
        )
        out = extract_episode_text(_payload([{"role": "user", "content": envelope}]))
        assert out == "Q: Actual user question?"

    def test_assistant_envelope_NOT_stripped(self):
        # Only user messages get envelope stripping (matches TS).
        text = (
            "Sender (untrusted metadata): ```json\n{}\n```\nstill here"
        )
        out = extract_episode_text(_payload([{"role": "assistant", "content": text}]))
        assert out.startswith("A: Sender")
