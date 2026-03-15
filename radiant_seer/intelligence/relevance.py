"""Headline-to-contract relevance scoring.

Combines domain knowledge (causal category graph) with a learned MLP
scorer and recency weighting to determine which headlines matter for
which contracts.

"Iraq embassy missile" → tags {conflict, middle_east}
"Oil above $85?"      → tags {energy}
conflict → energy is a causal edge → relevant!
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# ── Causal category definitions ──────────────────────────────────────
# Seeded from topic_filter.py patterns

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "conflict": [
        r"\bwar\b", r"\bmissile\b", r"\bairstrike\b", r"\bdrone strike\b",
        r"\binvasion\b", r"\bmilitary\b", r"\bstrike\b", r"\battack\b",
        r"\bwarship\b", r"\bnavy\b", r"\bIDF\b", r"\bcombat\b",
        r"\bbomb(?:ing|ed|s)?\b", r"\bshelling\b", r"\bceasefire\b",
        r"\bescalat(?:e|ion|ing)\b",
    ],
    "energy": [
        r"\boil\b", r"\bcrude\b", r"\bBrent\b", r"\bWTI\b",
        r"\bOPEC\b", r"\bpipeline\b", r"\bLNG\b", r"\bnatural gas\b",
        r"\bpetroleum\b", r"\brefiner(?:y|ies)\b", r"\btanker\b",
        r"\bbarrel[s]?\b", r"\benergy\b", r"\bfuel\b", r"\bgasoline\b",
        r"\bAramco\b",
    ],
    "sanctions": [
        r"\bsanction(?:s|ed|ing)?\b", r"\bembargo\b", r"\bblockade\b",
        r"\bfreez(?:e|ing)\s+assets\b", r"\btrade\s+restriction\b",
    ],
    "diplomacy": [
        r"\bdiploma(?:cy|tic)\b", r"\bnegotiat(?:e|ion|ing)\b",
        r"\bceasefire\b", r"\bJCPOA\b", r"\bnuclear deal\b",
        r"\bState Department\b", r"\bambassador\b", r"\btreaty\b",
        r"\bpeace\s+(?:deal|talk|process)\b",
    ],
    "middle_east": [
        r"\bIran\b", r"\bIranian\b", r"\bIraq\b", r"\bSaudi\b",
        r"\bHormuz\b", r"\bHouthi\b", r"\bHezbollah\b", r"\bHamas\b",
        r"\bGaza\b", r"\bLebanon\b", r"\bSyria\b", r"\bYemen\b",
        r"\bGulf\b", r"\bTehran\b", r"\bKhamenei\b", r"\bIRGC\b",
        r"\bKharg\b", r"\bIsrael\b", r"\bNetanyahu\b",
        r"\bRed Sea\b", r"\bPersian Gulf\b",
    ],
    "nuclear": [
        r"\buranium\b", r"\benrich(?:ment|ing|ed)\b",
        r"\bcentrifuge\b", r"\bIAEA\b", r"\bNatanz\b", r"\bFordow\b",
        r"\bnuclear\s+(?:weapon|program|facility|site)\b",
        r"\batomic\b", r"\bwarhead\b",
    ],
    "economics": [
        r"\bFed\b", r"\bFederal Reserve\b", r"\binterest rate\b",
        r"\binflation\b", r"\brecession\b", r"\bGDP\b",
        r"\bunemployment\b", r"\bCPI\b", r"\bS&P\s*500\b",
        r"\bmarket\s+crash\b", r"\bstock\s+market\b",
    ],
    "trade": [
        r"\btariff\b", r"\btrade war\b", r"\bcommodit(?:y|ies)\b",
        r"\bexport\b", r"\bimport\b", r"\btrade\s+deal\b",
        r"\btrade\s+deficit\b",
    ],
    "elections": [
        r"\belection\b", r"\bvot(?:e|ing|er)\b", r"\bpoll(?:s|ing)\b",
        r"\bTrump\b", r"\bBiden\b", r"\bcampaign\b",
        r"\bpresident(?:ial)?\b",
    ],
    "regime_change": [
        r"\bregime\b", r"\bcoup\b", r"\boverthrow\b",
        r"\bprotest(?:s|ers|ing)?\b", r"\buprising\b",
        r"\bdefect(?:ion|ed|ing)\b", r"\bdesert(?:ion|ed|ing)\b",
    ],
}

# Causal adjacency: which categories causally influence each other
CAUSAL_EDGES: dict[str, set[str]] = {
    "conflict": {"energy", "sanctions", "diplomacy", "middle_east", "regime_change"},
    "energy": {"conflict", "sanctions", "economics", "middle_east", "trade"},
    "sanctions": {"conflict", "energy", "diplomacy", "trade", "economics"},
    "diplomacy": {"conflict", "nuclear", "sanctions", "middle_east"},
    "middle_east": {"conflict", "energy", "nuclear", "diplomacy", "regime_change"},
    "nuclear": {"conflict", "middle_east", "sanctions", "diplomacy"},
    "economics": {"energy", "trade", "elections"},
    "trade": {"economics", "sanctions", "energy"},
    "elections": {"economics", "diplomacy", "regime_change"},
    "regime_change": {"conflict", "diplomacy", "elections", "middle_east"},
}

# Pre-compile all patterns
_COMPILED_PATTERNS: dict[str, re.Pattern] = {
    cat: re.compile("|".join(patterns), re.IGNORECASE)
    for cat, patterns in CATEGORY_PATTERNS.items()
}


class CausalDomainGraph:
    """Tag text with causal categories and compute domain relevance."""

    def tag(self, text: str) -> set[str]:
        """Return set of category tags for a text string."""
        tags = set()
        for cat, pattern in _COMPILED_PATTERNS.items():
            if pattern.search(text):
                tags.add(cat)
        return tags

    def tag_batch(self, texts: list[str]) -> list[set[str]]:
        """Tag a batch of texts."""
        return [self.tag(t) for t in texts]

    def relevance_score(
        self, headline_tags: set[str], contract_tags: set[str]
    ) -> float:
        """Compute domain relevance between headline and contract tags.

        Returns:
            Score in [0, 1]. Higher = more relevant.
        """
        if not headline_tags or not contract_tags:
            return 0.0

        # Direct tag overlap
        overlap = len(headline_tags & contract_tags)
        max_tags = max(len(headline_tags), len(contract_tags))
        direct_score = overlap / max_tags

        # Causal adjacency (1 hop)
        causal_links = 0
        for ht in headline_tags:
            edges = CAUSAL_EDGES.get(ht, set())
            causal_links += len(edges & contract_tags)
        causal_score = min(1.0, causal_links / 3)

        return 0.6 * direct_score + 0.4 * causal_score


class LearnedRelevanceScorer(nn.Module):
    """MLP that learns headline-to-contract relevance from prediction outcomes.

    Input: concat(headline_emb, contract_emb) = (emb_dim*2,)
    Output: relevance score in [0, 1]

    Trained end-to-end: gradients flow from prediction loss through
    attention weights back into this scorer.
    """

    def __init__(self, emb_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, headline_emb: Tensor, contract_emb: Tensor
    ) -> Tensor:
        """Score relevance for one contract against N headlines.

        Args:
            headline_emb: (N, emb_dim) headline embeddings.
            contract_emb: (emb_dim,) single contract embedding.

        Returns:
            scores: (N,) relevance scores in [0, 1].
        """
        N = headline_emb.shape[0]
        contract_expanded = contract_emb.unsqueeze(0).expand(N, -1)
        combined = torch.cat([headline_emb, contract_expanded], dim=-1)
        return self.net(combined).squeeze(-1)

    def batch_score(
        self, headline_embs: Tensor, contract_embs: Tensor
    ) -> Tensor:
        """Score all contracts against all headlines in one pass.

        Args:
            headline_embs: (N, emb_dim)
            contract_embs: (C, emb_dim)

        Returns:
            scores: (C, N) relevance matrix.
        """
        N = headline_embs.shape[0]
        C = contract_embs.shape[0]
        # Broadcast: (C, N, emb_dim) for each
        h_exp = headline_embs.unsqueeze(0).expand(C, -1, -1)
        c_exp = contract_embs.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([h_exp, c_exp], dim=-1)  # (C, N, emb_dim*2)
        return self.net(combined.reshape(C * N, -1)).reshape(C, N)


class RelevanceRouter:
    """Combines domain knowledge + learned relevance + recency weighting.

    For each (headline, contract) pair, produces a final attention weight
    that determines how much the headline influences this contract's prediction.
    """

    def __init__(
        self,
        domain_graph: CausalDomainGraph,
        learned_scorer: LearnedRelevanceScorer,
        alpha: float = 0.7,
        recency_halflife_hours: float = 6.0,
    ):
        self.domain_graph = domain_graph
        self.learned_scorer = learned_scorer
        self.alpha = alpha
        self.halflife = recency_halflife_hours
        self._decay_lambda = math.log(2) / max(recency_halflife_hours, 0.01)

    def _recency_weights(
        self, timestamps: list[float], now: float | None = None
    ) -> Tensor:
        """Exponential decay weights based on headline age."""
        now = now or time.time()
        ages_hours = [(now - ts) / 3600.0 for ts in timestamps]
        weights = [math.exp(-self._decay_lambda * max(0, age)) for age in ages_hours]
        return torch.tensor(weights, dtype=torch.float32)

    def cache_headline_tags(
        self, headline_texts: list[str]
    ) -> list[set[str]]:
        """Pre-compute headline tags once per cycle. Reuse across contracts."""
        return self.domain_graph.tag_batch(headline_texts)

    def compute_weights(
        self,
        headline_embs: Tensor,
        headline_timestamps: list[float],
        contract_emb: Tensor,
        headline_texts: list[str],
        contract_text: str,
        headline_tags: list[set[str]] | None = None,
    ) -> Tensor:
        """Compute attention weights for one contract over all headlines.

        Args:
            headline_embs: (N, emb_dim) headline embeddings.
            headline_timestamps: N unix timestamps.
            contract_emb: (emb_dim,) contract question embedding.
            headline_texts: N headline strings (for domain tagging).
            contract_text: Contract question string (for domain tagging).
            headline_tags: Pre-computed headline tags (from cache_headline_tags).

        Returns:
            weights: (N,) softmax-normalized attention weights.
        """
        N = len(headline_texts)
        if N == 0:
            return torch.zeros(0)

        device = headline_embs.device

        # 1. Recency
        recency = self._recency_weights(headline_timestamps).to(device)

        # 2. Domain graph scores (use cached tags if provided)
        if headline_tags is None:
            headline_tags = self.domain_graph.tag_batch(headline_texts)
        contract_tags = self.domain_graph.tag(contract_text)
        domain_scores = torch.zeros(N, device=device)
        for i, h_tags in enumerate(headline_tags):
            domain_scores[i] = self.domain_graph.relevance_score(
                h_tags, contract_tags
            )

        # 3. Learned scores
        with torch.no_grad():
            learned_scores = self.learned_scorer(
                headline_embs, contract_emb
            )

        # 4. Combine: recency * (alpha * domain + (1-alpha) * learned)
        combined = recency * (
            self.alpha * domain_scores
            + (1 - self.alpha) * learned_scores
        )

        # Add small epsilon so softmax doesn't produce all zeros
        combined = combined + 1e-8

        # Softmax normalize
        return torch.softmax(combined, dim=0)

    def compute_weights_batch(
        self,
        headline_embs: Tensor,
        headline_timestamps: list[float],
        contract_embs: Tensor,
        headline_texts: list[str],
        contract_texts: list[str],
        headline_tags: list[set[str]] | None = None,
    ) -> Tensor:
        """Compute attention weights for all contracts at once.

        Args:
            headline_embs: (N, emb_dim)
            headline_timestamps: N timestamps
            contract_embs: (C, emb_dim)
            headline_texts: N headline strings
            contract_texts: C contract question strings
            headline_tags: Pre-computed headline tags (from cache_headline_tags).

        Returns:
            weights: (C, N) attention weight matrix.
        """
        N = len(headline_texts)
        C = len(contract_texts)
        device = headline_embs.device

        if N == 0:
            return torch.zeros(C, 0, device=device)

        # 1. Recency (shared across all contracts)
        recency = self._recency_weights(headline_timestamps).to(device)

        # 2. Domain scores (C, N) — headline tags computed once
        if headline_tags is None:
            headline_tags = self.domain_graph.tag_batch(headline_texts)
        contract_tags_list = [self.domain_graph.tag(ct) for ct in contract_texts]
        domain_scores = torch.zeros(C, N, device=device)
        for c, c_tags in enumerate(contract_tags_list):
            for h, h_tags in enumerate(headline_tags):
                domain_scores[c, h] = self.domain_graph.relevance_score(
                    h_tags, c_tags
                )

        # 3. Learned scores (C, N)
        with torch.no_grad():
            learned_scores = self.learned_scorer.batch_score(
                headline_embs, contract_embs
            )

        # 4. Combine
        recency_expanded = recency.unsqueeze(0).expand(C, -1)
        combined = recency_expanded * (
            self.alpha * domain_scores
            + (1 - self.alpha) * learned_scores
        )
        combined = combined + 1e-8

        # Softmax per contract (across headlines)
        return torch.softmax(combined, dim=1)
