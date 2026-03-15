"""Fine-tune encoder and predictor on collected real data snapshots.

Loads snapshots from data/snapshots/, builds temporal pairs from
consecutive snapshots, and retrains with VICReg.

Usage:
    python -m radiant_seer.training.finetune_real
    python -m radiant_seer.training.finetune_real --epochs 50 --min-snapshots 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.collector import load_snapshots
from radiant_seer.intelligence.loss_functions import VICRegLoss
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.main_seer import OODDetector


class RealPairDataset(Dataset):
    """Pairs of consecutive real-world snapshots for VICReg training."""

    def __init__(self, snapshots: list[dict]):
        self.pairs: list[tuple[dict, dict]] = []
        for i in range(len(snapshots) - 1):
            s_t = snapshots[i]
            s_t1 = snapshots[i + 1]
            # Only pair snapshots that both have valid data
            if s_t.get("n_headlines", 0) > 0 and s_t1.get("n_headlines", 0) > 0:
                self.pairs.append((s_t, s_t1))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s_t, s_t1 = self.pairs[idx]
        return {
            "news_t": torch.tensor(s_t["news_embedding"], dtype=torch.float32),
            "macro_t": torch.tensor(s_t["macro_values"], dtype=torch.float32),
            "sentiment_t": torch.tensor(
                [s_t.get("sentiment", 0.0)], dtype=torch.float32
            ),
            "news_t1": torch.tensor(
                s_t1["news_embedding"], dtype=torch.float32
            ),
            "macro_t1": torch.tensor(
                s_t1["macro_values"], dtype=torch.float32
            ),
            "sentiment_t1": torch.tensor(
                [s_t1.get("sentiment", 0.0)], dtype=torch.float32
            ),
        }


def finetune_encoder(
    config: SeerConfig | None = None,
    snapshot_dir: Path | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    min_snapshots: int = 10,
) -> tuple[MultimodalEncoder, dict] | None:
    """Fine-tune the encoder on real data snapshots.

    Returns:
        Tuple of (fine-tuned encoder, history) or None if insufficient data.
    """
    config = config or SeerConfig()
    device = config.device

    # Load snapshots
    snapshots = load_snapshots(snapshot_dir)
    logger.info(f"Loaded {len(snapshots)} snapshots")

    if len(snapshots) < min_snapshots:
        logger.warning(
            f"Only {len(snapshots)} snapshots (need {min_snapshots}). "
            f"Collect more with: python -m radiant_seer.data_swarm.collector --loop 900"
        )
        return None

    # Build dataset
    dataset = RealPairDataset(snapshots)
    if len(dataset) < 2:
        logger.warning(f"Only {len(dataset)} valid pairs. Need at least 2.")
        return None

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
    )
    logger.info(f"Training on {len(dataset)} real pairs, {len(loader)} batches")

    # Load pre-trained encoder
    model_dir = config.project_root / "data" / "models"
    encoder = MultimodalEncoder(
        news_dim=config.news_embedding_dim,
        macro_dim=config.macro_feature_count,
        latent_dim=config.latent_dim,
    ).to(device)

    encoder_path = model_dir / "encoder_vicreg.pt"
    if encoder_path.exists():
        encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        logger.info("Loaded pre-trained encoder weights for fine-tuning")

    # Fine-tune with lower LR
    vicreg = VICRegLoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    history = {"loss": [], "invariance": [], "variance": [], "covariance": []}

    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0.0
        epoch_metrics = {"invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        n_batches = 0

        for batch in loader:
            z_t = encoder(
                batch["news_t"].to(device),
                batch["macro_t"].to(device),
                batch["sentiment_t"].to(device),
            )
            z_t1 = encoder(
                batch["news_t1"].to(device),
                batch["macro_t1"].to(device),
                batch["sentiment_t1"].to(device),
            )

            result = vicreg(z_t, z_t1)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for k in epoch_metrics:
                epoch_metrics[k] += result[k].item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches
        history["loss"].append(epoch_loss)
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
            history[k].append(epoch_metrics[k])

        if epoch % 10 == 0 or epoch == epochs - 1:
            encoder.eval()
            with torch.no_grad():
                sample = next(iter(loader))
                z = encoder(
                    sample["news_t"].to(device),
                    sample["macro_t"].to(device),
                    sample["sentiment_t"].to(device),
                )
                z_std = z.std(dim=0)
                min_std = z_std.min().item()

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={epoch_loss:.4f} | z_std min={min_std:.3f}"
            )

    # Save fine-tuned model
    save_path = model_dir / "encoder_finetuned.pt"
    torch.save(encoder.state_dict(), save_path)
    logger.info(f"Fine-tuned encoder saved: {save_path}")

    # Refit OOD baseline on real data embeddings
    encoder.eval()
    logger.info("Refitting OOD baseline on real data...")
    all_z = []
    with torch.no_grad():
        for snap in snapshots:
            if snap.get("n_headlines", 0) > 0:
                z = encoder(
                    torch.tensor(
                        [snap["news_embedding"]], dtype=torch.float32
                    ).to(device),
                    torch.tensor(
                        [snap["macro_values"]], dtype=torch.float32
                    ).to(device),
                    torch.tensor(
                        [[snap.get("sentiment", 0.0)]], dtype=torch.float32
                    ).to(device),
                )
                all_z.append(z)

    if all_z:
        z_train = torch.cat(all_z, dim=0)
        ood = OODDetector(threshold=config.mahalanobis_threshold)
        ood.fit(z_train)
        ood_path = model_dir / "ood_baseline_real.pt"
        torch.save(
            {"mean": ood._mean, "inv_cov": ood._inv_cov, "n_samples": len(all_z)},
            ood_path,
        )
        logger.info(f"Real OOD baseline saved: {ood_path} ({len(all_z)} states)")

    return encoder, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune on real data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-snapshots", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    finetune_encoder(
        epochs=args.epochs,
        lr=args.lr,
        min_snapshots=args.min_snapshots,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
