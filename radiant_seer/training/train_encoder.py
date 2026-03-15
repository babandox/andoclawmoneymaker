"""Phase 1 training: Multimodal Encoder with VICReg.

Trains on pairs of consecutive synthetic states (state_t, state_t+1).
VICReg ensures the latent space doesn't collapse while learning useful structure.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.synthetic import CivStateDataset, SyntheticCivStateGenerator
from radiant_seer.intelligence.loss_functions import VICRegLoss
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder


class PairDataset(torch.utils.data.Dataset):
    """Wraps CivStateDataset to yield (state_t, state_t+1) pairs for VICReg."""

    def __init__(self, dataset: CivStateDataset):
        self.dataset = dataset
        # Flatten episodes into consecutive pairs
        self._pairs: list[tuple[int, int]] = []  # (episode_idx, time_idx)
        for ep in range(len(dataset)):
            T = dataset.news.shape[1]
            for t in range(T - 1):
                self._pairs.append((ep, t))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep, t = self._pairs[idx]
        return {
            "news_t": self.dataset.news[ep, t],
            "macro_t": self.dataset.macro[ep, t],
            "sentiment_t": self.dataset.sentiment[ep, t],
            "news_t1": self.dataset.news[ep, t + 1],
            "macro_t1": self.dataset.macro[ep, t + 1],
            "sentiment_t1": self.dataset.sentiment[ep, t + 1],
            "regime_t": self.dataset.regimes[ep, t],
            "regime_t1": self.dataset.regimes[ep, t + 1],
        }


def train_encoder(
    config: SeerConfig | None = None,
    n_episodes: int = 500,
    episode_length: int = 50,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    save_path: Path | None = None,
) -> tuple[MultimodalEncoder, dict]:
    """Train the multimodal encoder with VICReg on synthetic data.

    Returns:
        Tuple of (trained encoder, training metrics dict).
    """
    config = config or SeerConfig()
    device = config.device
    logger.info(f"Training encoder on {device} | {n_episodes} episodes x {episode_length} steps")

    # Generate synthetic data
    logger.info("Generating synthetic dataset...")
    gen = SyntheticCivStateGenerator(seed=42)
    dataset = gen.generate_dataset(n_episodes=n_episodes, episode_length=episode_length)
    pair_dataset = PairDataset(dataset)
    loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    logger.info(f"Dataset: {len(pair_dataset)} pairs, {len(loader)} batches/epoch")

    # Model + loss + optimizer
    encoder = MultimodalEncoder(
        news_dim=config.news_embedding_dim,
        macro_dim=config.macro_feature_count,
        latent_dim=config.latent_dim,
    ).to(device)
    vicreg = VICRegLoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder params: {param_count:,}")

    # Training loop
    history = {"loss": [], "invariance": [], "variance": [], "covariance": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        encoder.train()
        epoch_metrics = {"loss": 0.0, "invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        n_batches = 0

        for batch in loader:
            news_t = batch["news_t"].to(device)
            macro_t = batch["macro_t"].to(device)
            sent_t = batch["sentiment_t"].to(device)
            news_t1 = batch["news_t1"].to(device)
            macro_t1 = batch["macro_t1"].to(device)
            sent_t1 = batch["sentiment_t1"].to(device)

            z_t = encoder(news_t, macro_t, sent_t)
            z_t1 = encoder(news_t1, macro_t1, sent_t1)

            result = vicreg(z_t, z_t1)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            for key in epoch_metrics:
                epoch_metrics[key] += result[key].item()
            n_batches += 1

        scheduler.step()

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            history[key].append(epoch_metrics[key])

        # Check for dimensional collapse
        encoder.eval()
        with torch.no_grad():
            sample = next(iter(loader))
            z_sample = encoder(
                sample["news_t"].to(device),
                sample["macro_t"].to(device),
                sample["sentiment_t"].to(device),
            )
            z_std = z_sample.std(dim=0)
            min_std = z_std.min().item()
            mean_std = z_std.mean().item()
            collapsed_dims = (z_std < 0.1).sum().item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={epoch_metrics['loss']:.4f} "
                f"inv={epoch_metrics['invariance']:.4f} "
                f"var={epoch_metrics['variance']:.4f} "
                f"cov={epoch_metrics['covariance']:.4f} | "
                f"z_std: min={min_std:.3f} mean={mean_std:.3f} collapsed={collapsed_dims}"
            )

        # Save best
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            if save_path:
                torch.save(encoder.state_dict(), save_path)

    # Final collapse check
    encoder.eval()
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(
        f"Final z_std: min={min_std:.3f}, mean={mean_std:.3f}, "
        f"collapsed_dims={collapsed_dims}/{config.latent_dim}"
    )

    if collapsed_dims > 0:
        logger.warning(f"DIMENSIONAL COLLAPSE: {collapsed_dims} dimensions have std < 0.1")
    else:
        logger.success("No dimensional collapse detected.")

    return encoder, history


if __name__ == "__main__":
    save_dir = Path("data/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    encoder, history = train_encoder(
        epochs=100,
        save_path=save_dir / "encoder_vicreg.pt",
    )
