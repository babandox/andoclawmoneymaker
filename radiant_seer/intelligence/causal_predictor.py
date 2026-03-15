"""Causal Predictor: learns the transition function z_{t+1} = f(z_t, event).

Intentionally an MLP (not RNN) — MCTS handles temporal chaining by
calling this repeatedly for multi-step rollouts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalPredictor(nn.Module):
    """Predict next latent state given current state and an event.

    Architecture:
        - Event embedding: nn.Embedding(num_event_types, latent_dim)
        - Predictor MLP: Linear(2*latent_dim → 256 → latent_dim) with GELU + LayerNorm
        - Input: cat([z_t, event_emb]) → predicted z_{t+1}
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_event_types: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_event_types = num_event_types

        self.event_embedding = nn.Embedding(num_event_types, latent_dim)

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Residual gate — learned blend of predicted delta and identity
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_t: Tensor, event: Tensor) -> Tensor:
        """Predict z_{t+1} from z_t and event type.

        Args:
            z_t: (batch, latent_dim) current latent state.
            event: (batch,) integer event type indices.

        Returns:
            z_next: (batch, latent_dim) predicted next latent state.
        """
        event_emb = self.event_embedding(event)  # (B, latent_dim)
        combined = torch.cat([z_t, event_emb], dim=-1)  # (B, 2*latent_dim)
        delta = self.predictor(combined)  # (B, latent_dim)

        # Gated residual: z_{t+1} = z_t + gate * delta
        g = self.gate(delta)
        z_next = z_t + g * delta

        return z_next

    def rollout(self, z_start: Tensor, events: Tensor) -> Tensor:
        """Multi-step rollout: chain predictions over a sequence of events.

        Args:
            z_start: (batch, latent_dim) starting latent state.
            events: (batch, num_steps) event type indices for each step.

        Returns:
            z_trajectory: (batch, num_steps + 1, latent_dim) including z_start.
        """
        B, T = events.shape
        trajectory = [z_start]
        z = z_start

        for t in range(T):
            z = self.forward(z, events[:, t])
            trajectory.append(z)

        return torch.stack(trajectory, dim=1)  # (B, T+1, latent_dim)
