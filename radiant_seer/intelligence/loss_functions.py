"""VICReg loss implementation per Bardes et al. (2022).

Three terms:
  - Invariance: MSE between embeddings of paired states
  - Variance: hinge loss ensuring std(z) >= 1 per dimension (prevents collapse)
  - Covariance: penalize off-diagonal covariance elements (decorrelation)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VICRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization.

    Args:
        lambda_var: Weight for variance term (default 25.0).
        mu_cov: Weight for covariance term (default 1.0).
        nu_inv: Weight for invariance term (default 25.0).
        variance_target: Target std per dimension (default 1.0).
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        lambda_var: float = 25.0,
        mu_cov: float = 1.0,
        nu_inv: float = 25.0,
        variance_target: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_var = lambda_var
        self.mu_cov = mu_cov
        self.nu_inv = nu_inv
        self.variance_target = variance_target
        self.eps = eps

    def invariance_loss(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """MSE between paired embeddings."""
        return F.mse_loss(z_a, z_b)

    def variance_loss(self, z: Tensor) -> Tensor:
        """Hinge loss on per-dimension std — prevents dimensional collapse."""
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(self.variance_target - std))

    def covariance_loss(self, z: Tensor) -> Tensor:
        """Penalize off-diagonal covariance — decorrelates dimensions."""
        batch_size, dim = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (batch_size - 1)

        # Zero diagonal, penalize off-diagonal
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / dim

    def forward(self, z_a: Tensor, z_b: Tensor) -> dict[str, Tensor]:
        """Compute VICReg loss.

        Args:
            z_a: First set of embeddings (batch_size, latent_dim).
            z_b: Second set of embeddings (batch_size, latent_dim).

        Returns:
            Dict with 'loss', 'invariance', 'variance', 'covariance' terms.
        """
        inv_loss = self.invariance_loss(z_a, z_b)

        var_loss_a = self.variance_loss(z_a)
        var_loss_b = self.variance_loss(z_b)
        var_loss = (var_loss_a + var_loss_b) / 2

        cov_loss_a = self.covariance_loss(z_a)
        cov_loss_b = self.covariance_loss(z_b)
        cov_loss = (cov_loss_a + cov_loss_b) / 2

        total = self.nu_inv * inv_loss + self.lambda_var * var_loss + self.mu_cov * cov_loss

        return {
            "loss": total,
            "invariance": inv_loss,
            "variance": var_loss,
            "covariance": cov_loss,
        }
