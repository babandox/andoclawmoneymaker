"""Multimodal encoder: fuses news embeddings, macro tensors, and sentiment
into a 128-dim latent vector z via modality-specific projections + Transformer
cross-attention fusion.

Architecture:
  - news_proj:      Linear(384→256→128) with GELU + LayerNorm
  - macro_proj:     Linear(12→64→128)
  - sentiment_proj: Linear(1→32→128)
  - fusion:         2-layer TransformerEncoder (d_model=128, nhead=8)
  - Output:         mean-pooled 128-dim latent vector z
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModalityProjection(nn.Module):
    """Two-layer projection head with GELU + LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultimodalEncoder(nn.Module):
    """Encode (news_emb, macro, sentiment) → 128-dim latent z."""

    def __init__(
        self,
        news_dim: int = 384,
        macro_dim: int = 12,
        sentiment_dim: int = 1,
        latent_dim: int = 128,
        nhead: int = 8,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Modality-specific projection heads
        self.news_proj = ModalityProjection(news_dim, 256, latent_dim)
        self.macro_proj = ModalityProjection(macro_dim, 64, latent_dim)
        self.sentiment_proj = ModalityProjection(sentiment_dim, 32, latent_dim)

        # Modality type embeddings (learned)
        self.modality_embeddings = nn.Embedding(3, latent_dim)

        # Transformer fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # Output projection (post-fusion normalization)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, news: Tensor, macro: Tensor, sentiment: Tensor
    ) -> Tensor:
        """Encode multimodal inputs to latent vector z.

        Args:
            news: (batch, news_dim) pre-computed sentence embeddings.
            macro: (batch, macro_dim) macro feature tensor.
            sentiment: (batch, 1) sentiment score.

        Returns:
            z: (batch, latent_dim) latent state vector.
        """
        # Project each modality to latent_dim
        news_z = self.news_proj(news)          # (B, latent_dim)
        macro_z = self.macro_proj(macro)        # (B, latent_dim)
        sent_z = self.sentiment_proj(sentiment) # (B, latent_dim)

        # Add modality type embeddings
        device = news.device
        news_z = news_z + self.modality_embeddings(torch.tensor(0, device=device))
        macro_z = macro_z + self.modality_embeddings(torch.tensor(1, device=device))
        sent_z = sent_z + self.modality_embeddings(torch.tensor(2, device=device))

        # Stack as sequence of 3 tokens: (B, 3, latent_dim)
        tokens = torch.stack([news_z, macro_z, sent_z], dim=1)

        # Transformer cross-attention fusion
        fused = self.fusion(tokens)  # (B, 3, latent_dim)

        # Mean pool across modality tokens
        z = fused.mean(dim=1)  # (B, latent_dim)
        z = self.output_norm(z)

        return z

    def forward_with_headlines(
        self, headlines: Tensor, macro: Tensor, sentiment: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode with per-headline outputs for contract-specific predictions.

        Args:
            headlines: (batch, N, news_dim) individual headline embeddings.
            macro: (batch, macro_dim) macro feature tensor.
            sentiment: (batch, 1) sentiment score.

        Returns:
            context_z: (batch, latent_dim) global context vector.
            headline_tokens: (batch, N, latent_dim) projected headline embeddings.
        """
        B, N, D = headlines.shape
        device = headlines.device

        # Project each headline individually through news_proj
        headlines_flat = headlines.reshape(B * N, D)
        headline_tokens = self.news_proj(headlines_flat).reshape(B, N, -1)

        # Mean-pool headlines for global context fusion
        news_mean = headline_tokens.mean(dim=1)  # (B, latent_dim)

        # Project macro and sentiment (same as forward)
        macro_z = self.macro_proj(macro)
        sent_z = self.sentiment_proj(sentiment)

        # Add modality embeddings
        news_z = news_mean + self.modality_embeddings(
            torch.tensor(0, device=device)
        )
        macro_z = macro_z + self.modality_embeddings(
            torch.tensor(1, device=device)
        )
        sent_z = sent_z + self.modality_embeddings(
            torch.tensor(2, device=device)
        )

        # Transformer fusion for global context
        tokens = torch.stack([news_z, macro_z, sent_z], dim=1)
        fused = self.fusion(tokens)
        context_z = self.output_norm(fused.mean(dim=1))

        return context_z, headline_tokens

    def encode_sequence(
        self, news: Tensor, macro: Tensor, sentiment: Tensor
    ) -> Tensor:
        """Encode a temporal sequence of states.

        Args:
            news: (batch, seq_len, news_dim)
            macro: (batch, seq_len, macro_dim)
            sentiment: (batch, seq_len, 1)

        Returns:
            z_seq: (batch, seq_len, latent_dim)
        """
        B, T, _ = news.shape
        # Flatten batch and time
        news_flat = news.reshape(B * T, -1)
        macro_flat = macro.reshape(B * T, -1)
        sent_flat = sentiment.reshape(B * T, -1)

        z_flat = self.forward(news_flat, macro_flat, sent_flat)
        return z_flat.reshape(B, T, self.latent_dim)
