"""News headline embedder using sentence-transformers.

Converts raw headline strings → 384-dim embeddings for the encoder.
Falls back to a simple TF-IDF-like hash if sentence-transformers isn't installed.
"""

from __future__ import annotations

import hashlib

import numpy as np
from loguru import logger


class NewsEmbedder:
    """Embed news headlines into fixed-dim vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        self._model = None
        self._model_name = model_name
        self._use_fallback = False

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info(f"NewsEmbedder: loaded {model_name}")
        except ImportError:
            self._use_fallback = True
            logger.warning(
                "sentence-transformers not installed — using hash fallback. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, headlines: list[str]) -> np.ndarray:
        """Embed a list of headlines.

        Args:
            headlines: List of headline strings.

        Returns:
            (N, dim) array of embeddings.
        """
        if not headlines:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._use_fallback:
            return self._hash_embed(headlines)

        embeddings = self._model.encode(
            headlines,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_aggregate(self, headlines: list[str]) -> np.ndarray:
        """Embed headlines and return a single mean-pooled vector.

        Args:
            headlines: List of headline strings.

        Returns:
            (dim,) mean-pooled embedding.
        """
        if not headlines:
            return np.zeros(self.dim, dtype=np.float32)

        embeddings = self.embed(headlines)
        mean_emb = embeddings.mean(axis=0)

        # L2 normalize
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        return mean_emb

    def embed_with_timestamps(
        self,
        headlines: list[str],
        timestamps: list[float] | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        """Embed headlines individually, preserving timestamps.

        Args:
            headlines: List of headline strings.
            timestamps: Optional unix timestamps per headline.
                If None, all get current time.

        Returns:
            embeddings: (N, dim) array of individual embeddings.
            timestamps: list of N timestamps.
        """
        if not headlines:
            return np.zeros((0, self.dim), dtype=np.float32), []

        embeddings = self.embed(headlines)

        if timestamps is None:
            import time as _time

            timestamps = [_time.time()] * len(headlines)

        return embeddings, timestamps

    def _hash_embed(self, headlines: list[str]) -> np.ndarray:
        """Deterministic fallback: hash-based pseudo-embeddings.

        NOT suitable for real use — only for testing the pipeline without
        installing sentence-transformers.
        """
        embeddings = np.zeros((len(headlines), self.dim), dtype=np.float32)
        for i, headline in enumerate(headlines):
            h = hashlib.sha256(headline.encode()).digest()
            # Expand hash to fill dim
            rng = np.random.RandomState(
                int.from_bytes(h[:4], "big")
            )
            embeddings[i] = rng.randn(self.dim).astype(np.float32)
            embeddings[i] /= np.linalg.norm(embeddings[i]) + 1e-8
        return embeddings
