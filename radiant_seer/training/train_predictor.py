"""Phase 2 training: Causal Predictor.

Learns z_{t+1} = f(z_t, event) using a frozen encoder to provide target latent states.
Trained with VICReg: predicted z_{t+1} should match encoded z_{t+1}.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.synthetic import CivStateDataset, SyntheticCivStateGenerator
from radiant_seer.intelligence.causal_predictor import CausalPredictor
from radiant_seer.intelligence.loss_functions import VICRegLoss
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder


class TransitionDataset(torch.utils.data.Dataset):
    """Yields (state_t, event_t, state_t+1) triples for predictor training."""

    def __init__(self, dataset: CivStateDataset):
        self.dataset = dataset
        self._triples: list[tuple[int, int]] = []
        for ep in range(len(dataset)):
            T = dataset.news.shape[1]
            for t in range(T - 1):
                self._triples.append((ep, t))

    def __len__(self) -> int:
        return len(self._triples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep, t = self._triples[idx]
        return {
            "news_t": self.dataset.news[ep, t],
            "macro_t": self.dataset.macro[ep, t],
            "sentiment_t": self.dataset.sentiment[ep, t],
            "news_t1": self.dataset.news[ep, t + 1],
            "macro_t1": self.dataset.macro[ep, t + 1],
            "sentiment_t1": self.dataset.sentiment[ep, t + 1],
            "event_t": self.dataset.events[ep, t],
        }


def train_predictor(
    encoder: MultimodalEncoder,
    config: SeerConfig | None = None,
    n_episodes: int = 500,
    episode_length: int = 50,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-4,
    save_path: Path | None = None,
) -> tuple[CausalPredictor, dict]:
    """Train the causal predictor with a frozen encoder.

    Args:
        encoder: Pre-trained multimodal encoder (will be frozen).

    Returns:
        Tuple of (trained predictor, training metrics dict).
    """
    config = config or SeerConfig()
    device = config.device
    logger.info(f"Training causal predictor on {device}")

    # Freeze encoder
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Generate data
    logger.info("Generating synthetic dataset...")
    gen = SyntheticCivStateGenerator(seed=123)
    dataset = gen.generate_dataset(n_episodes=n_episodes, episode_length=episode_length)
    trans_dataset = TransitionDataset(dataset)
    loader = DataLoader(trans_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    logger.info(f"Dataset: {len(trans_dataset)} transitions, {len(loader)} batches/epoch")

    # Model + loss + optimizer
    predictor = CausalPredictor(
        latent_dim=config.latent_dim,
        num_event_types=8,
    ).to(device)
    vicreg = VICRegLoss(lambda_var=10.0, mu_cov=1.0, nu_inv=25.0)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = sum(p.numel() for p in predictor.parameters())
    logger.info(f"Predictor params: {param_count:,}")

    # Training loop
    history = {"loss": [], "mse": [], "vicreg": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        predictor.train()
        epoch_mse = 0.0
        epoch_vicreg = 0.0
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            # Encode both states with frozen encoder
            with torch.no_grad():
                z_t = encoder(
                    batch["news_t"].to(device),
                    batch["macro_t"].to(device),
                    batch["sentiment_t"].to(device),
                )
                z_t1_target = encoder(
                    batch["news_t1"].to(device),
                    batch["macro_t1"].to(device),
                    batch["sentiment_t1"].to(device),
                )

            # Predict next state
            event = batch["event_t"].to(device)
            z_t1_pred = predictor(z_t, event)

            # Combined loss: MSE + VICReg
            l_mse = mse_loss(z_t1_pred, z_t1_target)
            vicreg_result = vicreg(z_t1_pred, z_t1_target)
            loss = l_mse + 0.1 * vicreg_result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            epoch_mse += l_mse.item()
            epoch_vicreg += vicreg_result["loss"].item()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        epoch_mse /= n_batches
        epoch_vicreg /= n_batches
        epoch_loss /= n_batches
        history["loss"].append(epoch_loss)
        history["mse"].append(epoch_mse)
        history["vicreg"].append(epoch_vicreg)

        if epoch % 10 == 0 or epoch == epochs - 1:
            # Check multi-step rollout stability
            predictor.eval()
            with torch.no_grad():
                z_start = z_t[:8]  # Take 8 samples from last batch
                events_5 = torch.randint(0, 8, (8, 5), device=device)
                trajectory = predictor.rollout(z_start, events_5)
                norms = trajectory.norm(dim=-1)
                norm_ratio = norms[:, -1].mean() / norms[:, 0].mean()

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={epoch_loss:.4f} mse={epoch_mse:.4f} vicreg={epoch_vicreg:.4f} | "
                f"5-step norm ratio={norm_ratio:.3f}"
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if save_path:
                torch.save(predictor.state_dict(), save_path)

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    return predictor, history


if __name__ == "__main__":
    from radiant_seer.training.train_encoder import train_encoder as _train_enc

    save_dir = Path("data/models")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Train encoder first
    encoder, _ = _train_enc(epochs=100, save_path=save_dir / "encoder_vicreg.pt")

    # Phase 2: Train predictor with frozen encoder
    predictor, _ = train_predictor(
        encoder, epochs=80, save_path=save_dir / "predictor.pt"
    )
