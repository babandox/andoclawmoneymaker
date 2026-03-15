"""State normalization: converts raw scraper output → state tensors the encoder expects.

Z-score normalization for macro features, pass-through for pre-computed embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class StateNormalizer:
    """Normalize heterogeneous data sources into encoder-ready tensors."""

    def __init__(self, macro_dim: int = 12, news_dim: int = 384):
        self.macro_dim = macro_dim
        self.news_dim = news_dim

        # Running statistics for macro z-score normalization
        self._macro_mean: np.ndarray | None = None
        self._macro_std: np.ndarray | None = None
        self._macro_count: int = 0

    def fit_macro(self, macro_data: np.ndarray) -> None:
        """Fit z-score parameters from historical macro data.

        Args:
            macro_data: (N, macro_dim) array of macro observations.
        """
        self._macro_mean = macro_data.mean(axis=0)
        self._macro_std = macro_data.std(axis=0)
        self._macro_std[self._macro_std < 1e-8] = 1.0  # Prevent division by zero
        self._macro_count = len(macro_data)

    def update_macro(self, new_obs: np.ndarray) -> None:
        """Online update of macro statistics with Welford's algorithm."""
        if self._macro_mean is None:
            self._macro_mean = new_obs.copy()
            self._macro_std = np.ones(self.macro_dim)
            self._macro_count = 1
            return

        self._macro_count += 1
        delta = new_obs - self._macro_mean
        self._macro_mean += delta / self._macro_count
        # Approximate std update
        self._macro_std = np.sqrt(
            ((self._macro_count - 1) * self._macro_std**2 + delta * (new_obs - self._macro_mean))
            / self._macro_count
        )
        self._macro_std[self._macro_std < 1e-8] = 1.0

    def normalize_macro(self, raw_macro: np.ndarray) -> Tensor:
        """Z-score normalize macro features.

        Args:
            raw_macro: (macro_dim,) or (batch, macro_dim) raw macro values.

        Returns:
            Normalized tensor.
        """
        if self._macro_mean is None:
            # No fit yet — return as-is
            return torch.tensor(raw_macro, dtype=torch.float32)

        normalized = (raw_macro - self._macro_mean) / self._macro_std
        return torch.tensor(normalized, dtype=torch.float32)

    def normalize_news(self, news_embedding: np.ndarray) -> Tensor:
        """Normalize news embedding (L2 normalization — standard for sentence embeddings).

        Args:
            news_embedding: (news_dim,) or (batch, news_dim) pre-computed embedding.

        Returns:
            L2-normalized tensor.
        """
        t = torch.tensor(news_embedding, dtype=torch.float32)
        if t.dim() == 1:
            norm = t.norm() + 1e-8
            return t / norm
        else:
            norms = t.norm(dim=-1, keepdim=True) + 1e-8
            return t / norms

    def normalize_sentiment(self, raw_sentiment: float) -> Tensor:
        """Clip and normalize sentiment to [-1, 1].

        Args:
            raw_sentiment: Raw sentiment score.

        Returns:
            (1,) tensor clipped to [-1, 1].
        """
        clipped = max(-1.0, min(1.0, raw_sentiment))
        return torch.tensor([clipped], dtype=torch.float32)

    def build_state(
        self,
        news_embedding: np.ndarray,
        macro_values: np.ndarray,
        sentiment: float,
    ) -> dict[str, Tensor]:
        """Build a complete state tensor dict from raw inputs.

        Returns:
            Dict with 'news', 'macro', 'sentiment' tensors ready for the encoder.
        """
        return {
            "news": self.normalize_news(news_embedding),
            "macro": self.normalize_macro(macro_values),
            "sentiment": self.normalize_sentiment(sentiment),
        }

    def build_state_v2(
        self,
        headline_embeddings: np.ndarray,
        headline_timestamps: list[float],
        headline_texts: list[str],
        macro_values: np.ndarray,
        sentiment: float,
    ) -> dict:
        """Build state with per-headline data for contract-specific predictions.

        Returns:
            Dict with:
              'headlines': Tensor (N, 384) — L2-normalized headline embeddings
              'headline_timestamps': list[float] — unix timestamps
              'headline_texts': list[str] — raw headline strings
              'news': Tensor (384,) — mean-pooled (backward compat)
              'macro': Tensor (12,)
              'sentiment': Tensor (1,)
        """
        headline_tensor = self.normalize_news(headline_embeddings)
        if headline_tensor.dim() == 1:
            headline_tensor = headline_tensor.unsqueeze(0)

        # Mean-pooled for backward compat
        if len(headline_embeddings) > 0:
            news_mean = headline_embeddings.mean(axis=0)
        else:
            news_mean = np.zeros(self.news_dim, dtype=np.float32)

        return {
            "headlines": headline_tensor,
            "headline_timestamps": headline_timestamps,
            "headline_texts": headline_texts,
            "news": self.normalize_news(news_mean),
            "macro": self.normalize_macro(macro_values),
            "sentiment": self.normalize_sentiment(sentiment),
        }
