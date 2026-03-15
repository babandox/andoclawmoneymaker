"""Vector DB: stores past event resolutions with their latent states.

Uses in-memory storage initially. ChromaDB backend will be added in Phase 4
when the 'data' optional dependencies are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch import Tensor


@dataclass
class SeerMemoryRecord:
    z_state: Tensor           # (latent_dim,) latent vector
    outcome: float            # Resolved outcome (0.0 or 1.0 for binary)
    contract_id: str = ""
    timestamp: datetime | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SimilarState:
    record: SeerMemoryRecord
    similarity: float


class SeerMemory:
    """Local vector store for historical event resolutions.

    Starts as pure-Python in-memory store. ChromaDB backend can be swapped in
    without changing the interface.
    """

    def __init__(self):
        self._records: list[SeerMemoryRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(
        self,
        z_state: Tensor,
        outcome: float,
        contract_id: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Store a resolved event with its latent state."""
        self._records.append(
            SeerMemoryRecord(
                z_state=z_state.detach().cpu(),
                outcome=outcome,
                contract_id=contract_id,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )
        )

    def find_similar_states(self, z_query: Tensor, k: int = 5) -> list[SimilarState]:
        """Find k most similar historical states by cosine similarity.

        Args:
            z_query: (latent_dim,) query latent vector.
            k: Number of neighbors.

        Returns:
            List of SimilarState sorted by descending similarity.
        """
        if not self._records:
            return []

        z_q = z_query.detach().cpu().float().flatten()
        z_q = z_q / (z_q.norm() + 1e-8)

        scored: list[tuple[float, SeerMemoryRecord]] = []
        for rec in self._records:
            z_r = rec.z_state.float().flatten()
            z_r = z_r / (z_r.norm() + 1e-8)
            sim = torch.dot(z_q, z_r).item()
            scored.append((sim, rec))

        scored.sort(key=lambda x: -x[0])
        return [
            SimilarState(record=rec, similarity=sim) for sim, rec in scored[:k]
        ]

    def get_outcome_prior(self, z_query: Tensor, k: int = 5) -> float:
        """Get a prior probability from similar historical outcomes.

        Returns 0.5 if no records exist.
        """
        neighbors = self.find_similar_states(z_query, k=k)
        if not neighbors:
            return 0.5

        # Similarity-weighted average of outcomes
        total_weight = sum(max(0, n.similarity) for n in neighbors)
        if total_weight < 1e-8:
            return 0.5

        weighted_outcome = sum(
            max(0, n.similarity) * n.record.outcome for n in neighbors
        )
        return weighted_outcome / total_weight
