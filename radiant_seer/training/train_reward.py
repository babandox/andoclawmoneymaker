"""Phase 3 training: Outcome Decoder (Reward Module).

Trains the small MLP that maps terminal latent states → P(contract_outcome).
Uses synthetic data where we know the ground truth outcome labels.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.synthetic import SyntheticCivStateGenerator
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.planning.reward_module import OutcomeDecoder


def _generate_outcome_labels(regimes: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Generate synthetic binary outcomes from regime + event data.

    Crisis regimes + high event types → higher probability of "YES" outcome
    for crisis-related contracts. This gives the decoder something meaningful to learn.
    """
    B, T = regimes.shape

    # Use terminal regime and event to determine outcome
    terminal_regime = regimes[:, -1].float()  # 0=normal, 1=volatile, 2=crisis
    terminal_event = events[:, -1].float()

    # P(YES) increases with regime severity and event type
    logit = -1.0 + 0.8 * terminal_regime + 0.15 * terminal_event
    prob = torch.sigmoid(logit)
    outcome = (torch.rand(B) < prob).float()
    return outcome


def train_reward_decoder(
    encoder: MultimodalEncoder,
    config: SeerConfig | None = None,
    n_episodes: int = 1000,
    episode_length: int = 50,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_path: Path | None = None,
) -> tuple[OutcomeDecoder, dict]:
    """Train the outcome decoder on synthetic labeled data.

    Args:
        encoder: Pre-trained multimodal encoder (frozen).

    Returns:
        Tuple of (trained decoder, training metrics dict).
    """
    config = config or SeerConfig()
    device = config.device
    logger.info(f"Training outcome decoder on {device}")

    # Freeze encoder
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Generate labeled data
    logger.info("Generating labeled synthetic dataset...")
    gen = SyntheticCivStateGenerator(seed=77)
    dataset = gen.generate_dataset(n_episodes=n_episodes, episode_length=episode_length)
    outcomes = _generate_outcome_labels(dataset.regimes, dataset.events)

    # Encode terminal states
    logger.info("Encoding terminal states...")
    with torch.no_grad():
        z_terminals = encoder(
            dataset.news[:, -1].to(device),
            dataset.macro[:, -1].to(device),
            dataset.sentiment[:, -1].to(device),
        ).cpu()

    # Split train/val
    n_train = int(0.8 * len(z_terminals))
    train_ds = TensorDataset(z_terminals[:n_train], outcomes[:n_train])
    val_ds = TensorDataset(z_terminals[n_train:], outcomes[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    logger.info(f"Train: {n_train}, Val: {len(z_terminals) - n_train}")

    # Model
    decoder = OutcomeDecoder(latent_dim=config.latent_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        decoder.train()
        train_loss = 0.0
        n_batches = 0
        for z_batch, y_batch in train_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            pred = decoder(z_batch).squeeze(-1)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Validate
        decoder.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for z_batch, y_batch in val_loader:
                z_batch, y_batch = z_batch.to(device), y_batch.to(device)
                pred = decoder(z_batch).squeeze(-1)
                val_loss += criterion(pred, y_batch).item()
                correct += ((pred > 0.5).float() == y_batch).sum().item()
                total += len(y_batch)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(decoder.state_dict(), save_path)

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    return decoder, history
