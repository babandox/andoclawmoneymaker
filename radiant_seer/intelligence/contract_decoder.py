"""Per-contract decoders: produce a unique probability for each contract.

V1 (ContractDecoder): Predicts from scratch using news + contract text.
V2 (ContractDecoderV2): Anchored on market price + prediction history.
    Takes the previous cycle's price as a starting point and learns
    to predict adjustments based on news and past accuracy.

Both are kept so they can be A/B tested side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class ContractHistory:
    """Per-contract state carried across cycles."""

    last_p_model: float = 0.5  # model's last prediction
    last_p_market: float = 0.5  # market price last cycle
    price_move: float = 0.0  # p_market_now - p_market_prev
    correct: float = 0.5  # 1.0 = right, 0.0 = wrong, 0.5 = unknown


class ContractDecoder(nn.Module):
    """V1: Predict from scratch using news + contract text.

    Architecture:
        1. Project contract embedding (384 → 128)
        2. Weighted sum of headline tokens using relevance weights → (128,)
        3. Concatenate [context_z, attended_headlines, contract_proj] → (384,)
        4. MLP → Sigmoid → per-contract probability

    Input shapes:
        context_z: (B, 128) — global context from encoder
        headline_tokens: (B, N, 128) — projected headline embeddings
        contract_emb: (B, 384) — contract question embedding
        relevance_weights: (B, N) — attention weights from RelevanceRouter

    Output: (B, 1) probability in [0, 1]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        contract_emb_dim: int = 384,
    ):
        super().__init__()
        self.contract_proj = nn.Sequential(
            nn.Linear(contract_emb_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

        # Small-variance init on final layer: predictions centered around 0.5
        # with enough spread to produce both BUY and SELL signals from the start.
        # LayerNorm upstream outputs ~N(0,1), so std=0.1 → pre-sigmoid range
        # of roughly [-0.3, 0.3] → Sigmoid outputs in [0.43, 0.57].
        nn.init.normal_(self.prediction_head[-2].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.prediction_head[-2].bias)

    def forward(
        self,
        context_z: Tensor,
        headline_tokens: Tensor,
        contract_emb: Tensor,
        relevance_weights: Tensor,
    ) -> Tensor:
        """Predict probability for one or more contracts.

        Args:
            context_z: (B, latent_dim) global context.
            headline_tokens: (B, N, latent_dim) per-headline embeddings.
            contract_emb: (B, contract_emb_dim) contract question embeddings.
            relevance_weights: (B, N) attention weights (should sum to ~1 per row).

        Returns:
            p: (B, 1) probability in [0, 1].
        """
        # 1. Project contract embedding to latent space
        contract_z = self.contract_proj(contract_emb)  # (B, latent_dim)

        # 2. Relevance-weighted attention over headlines
        # weights: (B, N, 1) for broadcasting
        w = relevance_weights.unsqueeze(-1)
        attended = (headline_tokens * w).sum(dim=1)  # (B, latent_dim)

        # 3. Concatenate and predict
        combined = torch.cat(
            [context_z, attended, contract_z], dim=-1
        )  # (B, latent_dim * 3)

        return self.prediction_head(combined)  # (B, 1)


class ContractDecoderV2(nn.Module):
    """V2: Anchored on market price — predicts corrections, not absolutes.

    Instead of predicting probability from scratch, this decoder outputs
    a small correction (delta) that gets added to the current market price.
    With random initial weights the delta is ~0, so predictions start at
    p_market and the model learns to deviate only when news warrants it.

    Market context features (4 scalars):
        p_market:     current market price (the anchor)
        prev_p_model: model's prediction last cycle (0.5 if first time)
        price_move:   price change since last prediction (0.0 if first time)
        prev_correct: was last prediction right? (0.5 = unknown)

    Architecture:
        1. Project contract embedding (384 → 128)
        2. Project market context (4 → 32)
        3. Weighted sum of headline tokens using relevance weights → (128,)
        4. Concatenate [context_z(128), attended(128), contract(128), market(32)] → (416,)
        5. MLP → Tanh → delta in [-1, 1] (scaled by max_delta=0.3)
        6. Output = clamp(p_market + delta, 0, 1)

    Output: (B, 1) probability in [0, 1]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        contract_emb_dim: int = 384,
        market_features: int = 4,
        market_hidden: int = 32,
        max_delta: float = 0.3,
    ):
        super().__init__()
        self.max_delta = max_delta
        self.contract_proj = nn.Sequential(
            nn.Linear(contract_emb_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.market_proj = nn.Sequential(
            nn.Linear(market_features, market_hidden),
            nn.GELU(),
            nn.LayerNorm(market_hidden),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim * 3 + market_hidden, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Tanh(),  # output in [-1, 1], scaled by max_delta
        )

        # Initialize delta_head final layer near zero so initial delta ≈ 0
        nn.init.zeros_(self.delta_head[-2].weight)
        nn.init.zeros_(self.delta_head[-2].bias)

    def forward(
        self,
        context_z: Tensor,
        headline_tokens: Tensor,
        contract_emb: Tensor,
        relevance_weights: Tensor,
        market_context: Tensor | None = None,
    ) -> Tensor:
        """Predict probability as p_market + learned correction.

        Args:
            context_z: (B, latent_dim) global context.
            headline_tokens: (B, N, latent_dim) per-headline embeddings.
            contract_emb: (B, contract_emb_dim) contract question embeddings.
            relevance_weights: (B, N) attention weights.
            market_context: (B, 4) [p_market, prev_p_model, price_move, correct].
                If None, uses defaults [0.5, 0.5, 0.0, 0.5].

        Returns:
            p: (B, 1) probability in [0, 1].
        """
        B = context_z.shape[0]
        device = context_z.device

        # 1. Project contract embedding
        contract_z = self.contract_proj(contract_emb)

        # 2. Project market context
        if market_context is None:
            market_context = torch.tensor(
                [[0.5, 0.5, 0.0, 0.5]], device=device
            ).expand(B, -1)
        market_z = self.market_proj(market_context)

        # 3. Relevance-weighted attention over headlines
        w = relevance_weights.unsqueeze(-1)
        attended = (headline_tokens * w).sum(dim=1)

        # 4. Predict delta (correction to market price)
        combined = torch.cat(
            [context_z, attended, contract_z, market_z], dim=-1
        )
        delta = self.delta_head(combined) * self.max_delta  # (B, 1) in [-0.3, 0.3]

        # 5. Apply correction to market price
        p_market = market_context[:, 0:1]  # (B, 1)
        return torch.clamp(p_market + delta, 0.0, 1.0)
