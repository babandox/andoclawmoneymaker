"""Content-based sentiment analysis using sentence-transformers.

Compares headline embeddings against positive/negative anchor phrases
via cosine similarity to determine actual tone of the news.

Replaces the broken upvote-ratio heuristic.
"""

from __future__ import annotations

import numpy as np

# Anchor phrases — averaged into positive/negative prototype embeddings
_POSITIVE_ANCHORS = [
    "peace deal signed, ceasefire holds, war ends",
    "oil prices drop, energy costs fall, fuel cheaper",
    "diplomatic breakthrough, negotiations succeed",
    "economy grows, markets rally, stability returns",
    "tensions ease, conflict de-escalates, troops withdraw",
    "sanctions lifted, trade resumes, cooperation",
]

_NEGATIVE_ANCHORS = [
    "war escalates, military strikes, bombing campaign",
    "oil prices surge, energy crisis, fuel shortage",
    "negotiations collapse, diplomacy fails, threats",
    "economy crashes, recession, markets plunge",
    "tensions rise, conflict intensifies, invasion",
    "sanctions tighten, embargo, blockade, seized",
]


class HeadlineSentimentAnalyzer:
    """Classify headline sentiment using embedding similarity."""

    def __init__(self, embedder):
        """Initialize with a NewsEmbedder instance."""
        self._embedder = embedder
        self._pos_proto: np.ndarray | None = None
        self._neg_proto: np.ndarray | None = None
        self._init_prototypes()

    def _init_prototypes(self) -> None:
        """Embed anchor phrases into prototype vectors."""
        pos_embs = self._embedder.embed(_POSITIVE_ANCHORS)
        neg_embs = self._embedder.embed(_NEGATIVE_ANCHORS)

        if len(pos_embs) > 0 and len(neg_embs) > 0:
            self._pos_proto = pos_embs.mean(axis=0)
            self._pos_proto /= np.linalg.norm(self._pos_proto) + 1e-8
            self._neg_proto = neg_embs.mean(axis=0)
            self._neg_proto /= np.linalg.norm(self._neg_proto) + 1e-8

    def score_headline(self, headline: str) -> float:
        """Score a single headline: -1 (bearish) to +1 (bullish).

        Computes cosine similarity to positive and negative prototypes,
        then returns the difference.
        """
        if self._pos_proto is None:
            return 0.0

        emb = self._embedder.embed([headline])
        if len(emb) == 0:
            return 0.0

        emb = emb[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        sim_pos = float(np.dot(emb, self._pos_proto))
        sim_neg = float(np.dot(emb, self._neg_proto))

        # Difference: positive = closer to peace/stability,
        # negative = closer to war/crisis
        raw = sim_pos - sim_neg
        # Scale to roughly [-1, 1] — typical diff range is [-0.3, 0.3]
        scaled = max(-1.0, min(1.0, raw * 3.0))
        return scaled

    def score_headlines(self, headlines: list[str]) -> float:
        """Score a batch of headlines, return average sentiment."""
        if not headlines or self._pos_proto is None:
            return 0.0

        embs = self._embedder.embed(headlines)
        if len(embs) == 0:
            return 0.0

        # Normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms

        sim_pos = embs @ self._pos_proto
        sim_neg = embs @ self._neg_proto

        diffs = sim_pos - sim_neg
        raw = float(diffs.mean())
        return max(-1.0, min(1.0, raw * 3.0))
