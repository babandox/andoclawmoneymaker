"""Tests for headline-to-contract relevance scoring."""

from __future__ import annotations

import time

import pytest
import torch

from radiant_seer.intelligence.relevance import (
    CausalDomainGraph,
    LearnedRelevanceScorer,
    RelevanceRouter,
)


# ── CausalDomainGraph ───────────────────────────────────────────────


class TestCausalDomainGraph:
    def setup_method(self):
        self.graph = CausalDomainGraph()

    def test_tag_conflict(self):
        tags = self.graph.tag("Iraq embassy hit by missile; no injuries")
        assert "conflict" in tags
        assert "middle_east" in tags

    def test_tag_energy(self):
        tags = self.graph.tag("Crude oil settles above $85 per barrel")
        assert "energy" in tags

    def test_tag_sanctions(self):
        tags = self.graph.tag("New sanctions imposed on Iranian oil exports")
        assert "sanctions" in tags
        assert "middle_east" in tags

    def test_tag_nuclear(self):
        tags = self.graph.tag("IAEA reports Iran enriching uranium to 60%")
        assert "nuclear" in tags
        assert "middle_east" in tags

    def test_tag_empty(self):
        tags = self.graph.tag("Taylor Swift concert in Los Angeles")
        assert len(tags) == 0

    def test_tag_multiple(self):
        tags = self.graph.tag(
            "US military strike on Iranian oil tanker near Hormuz"
        )
        assert "conflict" in tags
        assert "energy" in tags
        assert "middle_east" in tags

    def test_tag_batch(self):
        texts = ["Iran missile test", "Oil prices surge", "Cat video"]
        results = self.graph.tag_batch(texts)
        assert len(results) == 3
        assert "middle_east" in results[0]
        assert "energy" in results[1]
        assert len(results[2]) == 0

    def test_relevance_direct_overlap(self):
        """Same tags → high score."""
        score = self.graph.relevance_score(
            {"conflict", "middle_east"},
            {"conflict", "middle_east"},
        )
        assert score > 0.5

    def test_relevance_causal_link(self):
        """Conflict → energy causal edge → nonzero score."""
        score = self.graph.relevance_score(
            {"conflict", "middle_east"},
            {"energy"},
        )
        assert score > 0.0

    def test_relevance_no_connection(self):
        """No overlap, no causal link → zero."""
        score = self.graph.relevance_score(
            {"elections"},
            {"nuclear"},
        )
        assert score == 0.0

    def test_relevance_empty_tags(self):
        assert self.graph.relevance_score(set(), {"energy"}) == 0.0
        assert self.graph.relevance_score({"conflict"}, set()) == 0.0

    def test_iraq_missile_oil_relevance(self):
        """The key test: 'Iraq missile' should be relevant to oil contracts."""
        headline_tags = self.graph.tag(
            "Iraq embassy hit by missile; no injuries reported"
        )
        contract_tags = self.graph.tag(
            "Will crude oil price settle above $85 per barrel?"
        )
        score = self.graph.relevance_score(headline_tags, contract_tags)
        assert score > 0.1, f"Expected relevance > 0.1, got {score}"


# ── LearnedRelevanceScorer ───────────────────────────────────────────


class TestLearnedRelevanceScorer:
    def test_forward_shape(self):
        scorer = LearnedRelevanceScorer(emb_dim=384, hidden_dim=256)
        headline_embs = torch.randn(10, 384)
        contract_emb = torch.randn(384)
        scores = scorer(headline_embs, contract_emb)
        assert scores.shape == (10,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_batch_score_shape(self):
        scorer = LearnedRelevanceScorer(emb_dim=384, hidden_dim=256)
        headline_embs = torch.randn(20, 384)
        contract_embs = torch.randn(5, 384)
        scores = scorer.batch_score(headline_embs, contract_embs)
        assert scores.shape == (5, 20)

    def test_different_contracts_different_scores(self):
        """Different contracts should get different relevance distributions."""
        scorer = LearnedRelevanceScorer(emb_dim=384, hidden_dim=256)
        headlines = torch.randn(10, 384)
        c1 = torch.randn(384)
        c2 = torch.randn(384)
        s1 = scorer(headlines, c1)
        s2 = scorer(headlines, c2)
        # Very unlikely to be identical with random weights
        assert not torch.allclose(s1, s2, atol=1e-3)


# ── RelevanceRouter ──────────────────────────────────────────────────


class TestRelevanceRouter:
    def _make_router(self) -> RelevanceRouter:
        graph = CausalDomainGraph()
        scorer = LearnedRelevanceScorer(emb_dim=384, hidden_dim=256)
        return RelevanceRouter(
            domain_graph=graph,
            learned_scorer=scorer,
            alpha=0.7,
            recency_halflife_hours=6.0,
        )

    def test_compute_weights_shape(self):
        router = self._make_router()
        weights = router.compute_weights(
            headline_embs=torch.randn(10, 384),
            headline_timestamps=[time.time()] * 10,
            contract_emb=torch.randn(384),
            headline_texts=["headline"] * 10,
            contract_text="Will oil price rise?",
        )
        assert weights.shape == (10,)
        assert abs(weights.sum().item() - 1.0) < 1e-5  # softmax sums to 1

    def test_recency_weighting(self):
        """Recent headlines should get higher raw weights."""
        router = self._make_router()
        now = time.time()
        timestamps = [now, now - 3600, now - 86400]  # now, 1h ago, 24h ago
        weights = router._recency_weights(timestamps, now=now)
        assert weights[0] > weights[1] > weights[2]

    def test_relevant_headlines_weighted_higher(self):
        """Headlines matching contract topic should get higher weights."""
        router = self._make_router()
        now = time.time()
        texts = [
            "Iran launches missile at Iraq base",  # relevant to oil
            "Taylor Swift announces new album",  # irrelevant
        ]
        weights = router.compute_weights(
            headline_embs=torch.randn(2, 384),
            headline_timestamps=[now, now],
            contract_emb=torch.randn(384),
            headline_texts=texts,
            contract_text="Crude oil price above $85?",
        )
        # Iran missile should get higher weight than Taylor Swift for oil contract
        assert weights[0] > weights[1]

    def test_empty_headlines(self):
        router = self._make_router()
        weights = router.compute_weights(
            headline_embs=torch.zeros(0, 384),
            headline_timestamps=[],
            contract_emb=torch.randn(384),
            headline_texts=[],
            contract_text="test",
        )
        assert weights.shape == (0,)

    def test_batch_weights_shape(self):
        router = self._make_router()
        weights = router.compute_weights_batch(
            headline_embs=torch.randn(10, 384),
            headline_timestamps=[time.time()] * 10,
            contract_embs=torch.randn(5, 384),
            headline_texts=["headline"] * 10,
            contract_texts=["contract"] * 5,
        )
        assert weights.shape == (5, 10)
        # Each row should sum to ~1
        row_sums = weights.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(5), atol=1e-4)
