"""Cluster label generator. Port of TS `generateLabel` in packages/mindmap/src/labeler.ts."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from .mindmap_types import ConversationItem

# Verbatim from TS labeler.ts:4-18.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "like",
    "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "if", "when", "where", "how", "what", "which",
    "who", "whom", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their",
})

_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def extract_keywords(text: str) -> list[str]:
    """Lowercase, strip non-word, split, keep words >2 chars not in STOP_WORDS."""
    if not text:
        return []
    lowered = text.lower()
    no_punct = _NON_WORD_RE.sub(" ", lowered)
    words = _WS_RE.split(no_punct.strip())
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def generate_label(items: Iterable[ConversationItem], centroid: Sequence[float]) -> str:
    """Three-branch behavior matching TS labeler.ts:28-62."""
    items_list = list(items)

    if len(items_list) == 0:
        return "Empty"

    if len(items_list) == 1:
        words = extract_keywords(items_list[0].content)
        first_four = " ".join(words[:4])
        return first_four or items_list[0].content[:30]

    # Find item closest to centroid (ties: first-occurrence, matches TS `>` not `>=`)
    best_item = items_list[0]
    best_sim = -1.0
    for item in items_list:
        sim = _cosine(item.embedding, centroid)
        if sim > best_sim:
            best_sim = sim
            best_item = item

    representative = extract_keywords(best_item.content)
    if representative:
        return " ".join(representative[:4])

    # Fallback: top-3 most frequent across all items.
    freq: Counter[str] = Counter()
    for item in items_list:
        freq.update(extract_keywords(item.content))
    # Counter.most_common is stable; matches JS sort-by-frequency w/ insertion ordering.
    top_three = [w for w, _ in freq.most_common(3)]
    return " ".join(top_three) or "Cluster"


__all__ = ["STOP_WORDS", "extract_keywords", "generate_label"]
