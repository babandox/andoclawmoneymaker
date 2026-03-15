"""Reward module: maps terminal latent states to contract outcome probabilities.

This is where misplaced lines get detected:
  P_model  = outcome_decoder(z_terminal)  — what our model believes
  P_market = current exchange price        — what the market prices
  Reward   = P_model - P_market            — the edge/alpha
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from radiant_seer.intelligence.contract_decoder import ContractDecoder


@dataclass
class AlphaSignal:
    p_model: float
    p_market: float
    edge: float         # p_model - p_market
    abs_edge: float     # |edge|
    direction: str      # "BUY" or "SELL"


class OutcomeDecoder(nn.Module):
    """Small MLP mapping latent z → P(contract_outcome)."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Map latent state to probability.

        Args:
            z: (batch, latent_dim) or (latent_dim,) latent state.

        Returns:
            p: (batch, 1) or (1,) probability in [0, 1].
        """
        return self.net(z)


class RewardModule:
    """Compute reward signal as the edge between model and market probabilities."""

    def __init__(self, outcome_decoder: OutcomeDecoder):
        self.outcome_decoder = outcome_decoder

    @torch.no_grad()
    def compute_reward(self, z_terminal: Tensor, p_market: float) -> AlphaSignal:
        """Compute alpha signal for a single terminal state.

        Args:
            z_terminal: (latent_dim,) terminal latent state from MCTS rollout.
            p_market: Current market price for the contract.

        Returns:
            AlphaSignal with model probability, market price, and edge.
        """
        p_model = self.outcome_decoder(z_terminal.unsqueeze(0)).item()
        edge = p_model - p_market

        return AlphaSignal(
            p_model=p_model,
            p_market=p_market,
            edge=edge,
            abs_edge=abs(edge),
            direction="BUY" if edge > 0 else "SELL",
        )

    @torch.no_grad()
    def batch_rewards(self, z_terminals: Tensor, p_market: float) -> Tensor:
        """Compute rewards for a batch of terminal states.

        Args:
            z_terminals: (batch, latent_dim) terminal states.
            p_market: Current market price.

        Returns:
            rewards: (batch,) edge values.
        """
        p_models = self.outcome_decoder(z_terminals).squeeze(-1)  # (batch,)
        return p_models - p_market


class ContractRewardModule:
    """Per-contract reward computation using ContractDecoder."""

    def __init__(self, contract_decoder: ContractDecoder):
        self.contract_decoder = contract_decoder

    @torch.no_grad()
    def compute_reward(
        self,
        context_z: Tensor,
        headline_tokens: Tensor,
        contract_emb: Tensor,
        relevance_weights: Tensor,
        p_market: float,
    ) -> AlphaSignal:
        p_model = self.contract_decoder(
            context_z.unsqueeze(0),
            headline_tokens.unsqueeze(0),
            contract_emb.unsqueeze(0),
            relevance_weights.unsqueeze(0),
        ).item()
        edge = p_model - p_market
        return AlphaSignal(
            p_model=p_model,
            p_market=p_market,
            edge=edge,
            abs_edge=abs(edge),
            direction="BUY" if edge > 0 else "SELL",
        )
