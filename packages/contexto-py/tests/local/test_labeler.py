"""Tests for the local labeler — port of TS labeler.ts."""

from __future__ import annotations

from contexto_hermes.local.labeler import (
    STOP_WORDS,
    extract_keywords,
    generate_label,
)
from contexto_hermes.local.mindmap_types import ConversationItem


def _item(content: str, embedding=None) -> ConversationItem:
    return ConversationItem(
        id=content[:10] or "_",
        role="assistant",
        content=content,
        embedding=embedding or [1.0, 0.0],
    )


class TestExtractKeywords:
    def test_lowercases(self):
        assert extract_keywords("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert extract_keywords("foo, bar! baz?") == ["foo", "bar", "baz"]

    def test_filters_words_3_chars_or_less(self):
        assert extract_keywords("a be the cat") == ["cat"]

    def test_filters_stop_words(self):
        result = extract_keywords("the deployment failed")
        assert "the" not in result
        assert "deployment" in result
        assert "failed" in result

    def test_empty_string(self):
        assert extract_keywords("") == []


class TestStopWords:
    def test_classic_stop_words_present(self):
        for w in ("the", "and", "is", "have", "this", "that"):
            assert w in STOP_WORDS

    def test_keeps_meaningful_words(self):
        for w in ("deployment", "error", "kubernetes"):
            assert w not in STOP_WORDS


class TestGenerateLabel:
    def test_empty_items(self):
        assert generate_label([], [0.0, 0.0]) == "Empty"

    def test_single_item_uses_first_four_keywords(self):
        item = _item("kubernetes deployment failed with network error")
        assert generate_label([item], [1.0, 0.0]) == "kubernetes deployment failed network"

    def test_single_item_fallback_to_content_prefix(self):
        # All stop-words; keyword extraction returns nothing.
        item = _item("the and or but")
        label = generate_label([item], [1.0, 0.0])
        assert label == "the and or but"  # content[:30]

    def test_multiple_items_uses_centroid_nearest(self):
        # Two items; one matches the centroid direction.
        items = [
            _item("apple banana cherry date", embedding=[1.0, 0.0]),
            _item("kubernetes deployment errors failed", embedding=[0.0, 1.0]),
        ]
        centroid = [0.0, 1.0]
        label = generate_label(items, centroid)
        # The "kubernetes..." item is closer to centroid
        assert label == "kubernetes deployment errors failed"

    def test_multiple_items_fallback_to_top_frequency(self):
        # Two items whose nearest-centroid item has no keywords —
        # falls back to top-3 frequency across all items.
        items = [
            _item("the and or but", embedding=[1.0, 0.0]),  # centroid-nearest
            _item("deployment deployment errors", embedding=[0.0, 1.0]),
        ]
        centroid = [1.0, 0.0]
        label = generate_label(items, centroid)
        # Top-3 frequency: "deployment" (2), "errors" (1)
        assert "deployment" in label.split()
        assert "errors" in label.split()

    def test_multiple_items_final_fallback(self):
        # All items contain only stop-words / short tokens
        items = [
            _item("the a is", embedding=[1.0, 0.0]),
            _item("of in at", embedding=[0.0, 1.0]),
        ]
        assert generate_label(items, [1.0, 1.0]) == "Cluster"
