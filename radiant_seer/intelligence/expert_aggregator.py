"""Expert Aggregator: nearest-neighbor retrieval for MCTS prior estimation.

Minimal version: retrieves similar historical states from vector DB and
returns a prior probability for MCTS based on their outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AggregatorResult:
    prior: float
    confidence: float
    n_neighbors: int
    neighbor_outcomes: list[float]


class ExpertAggregator:
    """Aggregate historical outcomes from similar states to produce MCTS prior."""

    def __init__(self, memory_bank: MemoryBank | None = None):
        self._memory_bank = memory_bank

    @property
    def memory_bank(self) -> MemoryBank | None:
        return self._memory_bank

    @memory_bank.setter
    def memory_bank(self, bank: MemoryBank) -> None:
        self._memory_bank = bank

    def get_prior(self, z_query: Tensor, k: int = 5) -> AggregatorResult:
        """Get prior probability from similar historical states.

        Args:
            z_query: (latent_dim,) query latent vector.
            k: Number of neighbors to retrieve.

        Returns:
            AggregatorResult with prior estimate and confidence.
        """
        if self._memory_bank is None or len(self._memory_bank) == 0:
            return AggregatorResult(
                prior=0.5, confidence=0.0, n_neighbors=0, neighbor_outcomes=[]
            )

        neighbors = self._memory_bank.query(z_query, k=k)
        outcomes = [n.outcome for n in neighbors]

        prior = sum(outcomes) / len(outcomes)
        # Confidence based on agreement and number of neighbors
        if len(outcomes) > 1:
            variance = sum((o - prior) ** 2 for o in outcomes) / len(outcomes)
            confidence = max(0.0, 1.0 - variance) * min(1.0, len(outcomes) / k)
        else:
            confidence = 0.2

        return AggregatorResult(
            prior=prior,
            confidence=confidence,
            n_neighbors=len(outcomes),
            neighbor_outcomes=outcomes,
        )


@dataclass
class MemoryEntry:
    z_state: Tensor
    outcome: float  # 0.0 or 1.0 for binary contracts


class MemoryBank:
    """Simple in-memory nearest-neighbor store. Will be backed by ChromaDB in Phase 4."""

    def __init__(self):
        self.entries: list[MemoryEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, z_state: Tensor, outcome: float) -> None:
        self.entries.append(MemoryEntry(z_state=z_state.detach().cpu(), outcome=outcome))

    def query(self, z_query: Tensor, k: int = 5) -> list[MemoryEntry]:
        """Find k nearest neighbors by cosine similarity."""
        if not self.entries:
            return []

        z_q = z_query.detach().cpu().float()
        z_q = z_q / (z_q.norm() + 1e-8)

        similarities = []
        for entry in self.entries:
            z_e = entry.z_state.float()
            z_e = z_e / (z_e.norm() + 1e-8)
            sim = torch.dot(z_q.flatten(), z_e.flatten()).item()
            similarities.append(sim)

        top_k = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:k]
        return [self.entries[i] for i in top_k]
