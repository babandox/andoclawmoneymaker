"""Synthetic data generators for training and testing the intelligence core.

Produces temporal sequences of (news_embeddings, macro_tensors, sentiment, event_label)
with controlled causal structure: regime switches, correlated macro random walks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from torch.utils.data import Dataset


class Regime(IntEnum):
    NORMAL = 0
    VOLATILE = 1
    CRISIS = 2


@dataclass
class EpisodeConfig:
    news_dim: int = 384
    macro_dim: int = 12
    episode_length: int = 50
    regime_switch_prob: float = 0.05
    crisis_prob: float = 0.01
    num_event_types: int = 8


class CivStateDataset(Dataset):
    """Dataset of synthetic civilizational state sequences."""

    def __init__(
        self,
        news: torch.Tensor,       # (N, T, news_dim)
        macro: torch.Tensor,      # (N, T, macro_dim)
        sentiment: torch.Tensor,  # (N, T, 1)
        regimes: torch.Tensor,    # (N, T)
        events: torch.Tensor,     # (N, T)
    ):
        self.news = news
        self.macro = macro
        self.sentiment = sentiment
        self.regimes = regimes
        self.events = events

    def __len__(self) -> int:
        return self.news.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "news": self.news[idx],
            "macro": self.macro[idx],
            "sentiment": self.sentiment[idx],
            "regimes": self.regimes[idx],
            "events": self.events[idx],
        }


class SyntheticCivStateGenerator:
    """Generates synthetic civilizational state data with controlled causal structure."""

    def __init__(self, config: EpisodeConfig | None = None, seed: int = 42):
        self.config = config or EpisodeConfig()
        self.rng = np.random.default_rng(seed)

    def _generate_regime_sequence(self, length: int) -> np.ndarray:
        """Generate regime sequence with Markov transitions."""
        regimes = np.zeros(length, dtype=np.int64)
        regime = Regime.NORMAL

        # Transition matrix: rows=from, cols=to
        transition = np.array([
            [0.94, 0.05, 0.01],  # NORMAL →
            [0.10, 0.85, 0.05],  # VOLATILE →
            [0.15, 0.20, 0.65],  # CRISIS →
        ])

        for t in range(length):
            regimes[t] = regime
            regime = Regime(self.rng.choice(3, p=transition[regime]))

        return regimes

    def _generate_macro_walk(self, length: int, regimes: np.ndarray) -> np.ndarray:
        """Generate correlated macro random walks influenced by regime."""
        macro = np.zeros((length, self.config.macro_dim))

        # Regime-dependent volatility multipliers
        vol_mult = {Regime.NORMAL: 1.0, Regime.VOLATILE: 3.0, Regime.CRISIS: 6.0}

        # Base correlations between macro features (block structure)
        base_cov = np.eye(self.config.macro_dim)
        # Rates cluster (features 0-2)
        base_cov[0:3, 0:3] = 0.6
        # Inflation cluster (features 3-5)
        base_cov[3:6, 3:6] = 0.5
        # Employment cluster (features 6-8)
        base_cov[6:9, 6:9] = 0.4
        np.fill_diagonal(base_cov, 1.0)

        # Cross-cluster correlations
        base_cov[0:3, 3:6] = -0.2  # Rates vs inflation
        base_cov[3:6, 0:3] = -0.2

        # Initial values: centered, slightly random
        macro[0] = self.rng.standard_normal(self.config.macro_dim) * 0.5

        for t in range(1, length):
            vol = vol_mult[Regime(regimes[t])]
            noise = self.rng.multivariate_normal(
                np.zeros(self.config.macro_dim),
                base_cov * 0.01 * vol,
            )
            # Mean-reverting random walk
            macro[t] = macro[t - 1] * 0.99 + noise

        return macro

    def _generate_news_embeddings(
        self, length: int, regimes: np.ndarray, macro: np.ndarray
    ) -> np.ndarray:
        """Generate synthetic news embeddings correlated with regime and macro state."""
        news = np.zeros((length, self.config.news_dim))

        # Regime prototypes (different regions of embedding space)
        prototypes = {
            Regime.NORMAL: self.rng.standard_normal(self.config.news_dim) * 0.3,
            Regime.VOLATILE: self.rng.standard_normal(self.config.news_dim) * 0.5,
            Regime.CRISIS: self.rng.standard_normal(self.config.news_dim) * 0.8,
        }

        for t in range(length):
            regime = Regime(regimes[t])
            prototype = prototypes[regime]

            # Macro influence: project macro state into news embedding space
            macro_proj = np.zeros(self.config.news_dim)
            macro_proj[: self.config.macro_dim] = macro[t] * 0.1

            noise = self.rng.standard_normal(self.config.news_dim) * 0.2
            news[t] = prototype + macro_proj + noise

            # Normalize to unit sphere (like real sentence embeddings)
            norm = np.linalg.norm(news[t])
            if norm > 0:
                news[t] /= norm

        return news

    def _generate_sentiment(
        self, length: int, regimes: np.ndarray, macro: np.ndarray
    ) -> np.ndarray:
        """Generate sentiment signal correlated with regime and macro state."""
        sentiment = np.zeros((length, 1))

        regime_base = {Regime.NORMAL: 0.0, Regime.VOLATILE: -0.3, Regime.CRISIS: -0.7}

        for t in range(length):
            regime = Regime(regimes[t])
            base = regime_base[regime]
            macro_influence = np.mean(macro[t]) * 0.1
            noise = self.rng.standard_normal() * 0.15
            sentiment[t, 0] = np.clip(base + macro_influence + noise, -1.0, 1.0)

        return sentiment

    def _generate_events(self, length: int, regimes: np.ndarray) -> np.ndarray:
        """Generate event type sequence influenced by regime."""
        events = np.zeros(length, dtype=np.int64)

        for t in range(length):
            regime = Regime(regimes[t])
            if regime == Regime.CRISIS:
                # Crisis events (types 6-7) more likely
                weights = np.ones(self.config.num_event_types)
                weights[6:] = 5.0
            elif regime == Regime.VOLATILE:
                # Volatile events (types 4-5) more likely
                weights = np.ones(self.config.num_event_types)
                weights[4:6] = 3.0
            else:
                weights = np.ones(self.config.num_event_types)

            weights /= weights.sum()
            events[t] = self.rng.choice(self.config.num_event_types, p=weights)

        return events

    def generate_episode(self) -> dict[str, np.ndarray]:
        """Generate a single episode of civilizational state data."""
        length = self.config.episode_length
        regimes = self._generate_regime_sequence(length)
        macro = self._generate_macro_walk(length, regimes)
        news = self._generate_news_embeddings(length, regimes, macro)
        sentiment = self._generate_sentiment(length, regimes, macro)
        events = self._generate_events(length, regimes)

        return {
            "news": news,
            "macro": macro,
            "sentiment": sentiment,
            "regimes": regimes,
            "events": events,
        }

    def generate_dataset(
        self, n_episodes: int = 200, episode_length: int | None = None
    ) -> CivStateDataset:
        """Generate a full dataset of episodes."""
        if episode_length is not None:
            self.config.episode_length = episode_length

        all_news, all_macro, all_sentiment = [], [], []
        all_regimes, all_events = [], []

        for _ in range(n_episodes):
            ep = self.generate_episode()
            all_news.append(ep["news"])
            all_macro.append(ep["macro"])
            all_sentiment.append(ep["sentiment"])
            all_regimes.append(ep["regimes"])
            all_events.append(ep["events"])

        return CivStateDataset(
            news=torch.tensor(np.stack(all_news), dtype=torch.float32),
            macro=torch.tensor(np.stack(all_macro), dtype=torch.float32),
            sentiment=torch.tensor(np.stack(all_sentiment), dtype=torch.float32),
            regimes=torch.tensor(np.stack(all_regimes), dtype=torch.long),
            events=torch.tensor(np.stack(all_events), dtype=torch.long),
        )
