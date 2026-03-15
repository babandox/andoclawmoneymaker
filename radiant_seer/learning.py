"""Online learning from scored predictions — PnL-weighted policy gradient.

The model learns from PROFIT SIGNAL, not price regression:
  - Big win ($142 on crude oil) → strong positive gradient, be MORE confident
  - Small win ($0.50) → weak positive gradient
  - Loss (-$10) → negative gradient, be LESS confident there
  - Zero price movement → SKIP (no signal, no damage)

This teaches the model to chase asymmetric payoffs and avoid losers,
while anchor regularization prevents forgetting pre-trained knowledge.

The encoder stays frozen. The decoder + relevance scorer learn together.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor, nn

from radiant_seer.intelligence.contract_decoder import ContractDecoder
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.intelligence.relevance import (
    LearnedRelevanceScorer,
    RelevanceRouter,
)
from radiant_seer.planning.reward_module import OutcomeDecoder


@dataclass
class Experience:
    """One training sample: world state at prediction time + observed price."""

    news_emb: list[float]
    macro: list[float]
    sentiment: float
    target: float  # p_market_after — where the price actually went


class ReplayBuffer:
    """Fixed-size ring buffer of training experiences."""

    def __init__(self, maxlen: int = 10000):
        self.buffer: deque[Experience] = deque(maxlen=maxlen)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, n: int) -> list[Experience]:
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineLearner:
    """V1 online learner — FROZEN during live running.

    The outcome decoder maps a single world-state z to one probability,
    shared across all contracts. This architecture cannot learn per-contract
    patterns, and MSE training actively destroys pre-trained edge.

    Learning is disabled. The decoder is preserved as-is from pre-training.
    Use OnlineLearnerV2 for live learning.
    """

    def __init__(
        self,
        encoder: MultimodalEncoder,
        decoder: OutcomeDecoder,
        lr: float = 1e-4,
        batch_size: int = 64,
        steps_per_cycle: int = 5,
        device: str = "cpu",
        **_kwargs,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.batch_size = batch_size
        self.steps_per_cycle = steps_per_cycle

        self.replay = ReplayBuffer(maxlen=1_000_000)

        # Only train the decoder — encoder stays frozen
        self.optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Stats
        self.total_steps: int = 0
        self.recent_losses: deque[float] = deque(maxlen=100)

    def record_outcomes(
        self,
        news_emb: list[float],
        macro: list[float],
        sentiment: float,
        outcomes: list[float],
    ) -> int:
        """Record scored predictions — buffer only, no learning."""
        for target in outcomes:
            self.replay.add(
                Experience(
                    news_emb=news_emb,
                    macro=macro,
                    sentiment=sentiment,
                    target=target,
                )
            )
        return len(outcomes)

    @torch.no_grad()
    def _encode_batch(self, batch: list[Experience]) -> Tensor:
        """Encode a batch of experiences. Encoder is frozen."""
        news = torch.tensor(
            [e.news_emb for e in batch], dtype=torch.float32
        ).to(self.device)
        macro = torch.tensor(
            [e.macro for e in batch], dtype=torch.float32
        ).to(self.device)
        sent = torch.tensor(
            [[e.sentiment] for e in batch], dtype=torch.float32
        ).to(self.device)
        return self.encoder(news, macro, sent)

    def learn_step(self) -> float | None:
        """V1 learning is disabled — returns None without updating weights.

        The outcome decoder's pre-trained weights are preserved. Online MSE
        training was pushing p_model toward p_market, destroying edge.
        """
        return None

    def save_decoder(self, path: Path) -> None:
        """Save updated decoder weights to disk."""
        torch.save(self.decoder.state_dict(), path)
        logger.info(f"Saved decoder → {path}")

    @property
    def avg_loss(self) -> float | None:
        if not self.recent_losses:
            return None
        return sum(self.recent_losses) / len(self.recent_losses)

    @property
    def buffer_size(self) -> int:
        return len(self.replay)


# ── V2: Per-contract learning with PnL-weighted policy gradient ──────


@dataclass
class SharedCycleData:
    """Headline data shared across all experiences in one cycle."""

    headline_embs: list[list[float]]  # (N, 384) as nested list
    headline_timestamps: list[float]
    headline_texts: list[str]
    macro: list[float]
    sentiment: float
    cycle_id: int


@dataclass
class ExperienceV2:
    """Per-contract training sample with profit signal."""

    cycle_id: int
    contract_emb: list[float]  # (384,)
    contract_text: str
    target: float  # p_market_after (kept for compat)
    direction: str = "BUY"  # "BUY" or "SELL"
    pnl_dollars: float = 0.0  # dollar P&L — the training signal
    p_market: float = 0.5  # market price at prediction time
    liquidity: float = 0.0  # contract liquidity in USD


class OnlineLearnerV2:
    """Per-contract PnL-weighted policy gradient learning.

    Instead of MSE(p_model, p_market_after), uses dollar P&L as reward:
      - Won $142 on crude oil? Strong gradient: "be MORE confident here"
      - Lost $10 on gas? Gradient: "be LESS confident here"
      - Price didn't move? Skip entirely (no signal)

    The gradient magnitude is proportional to |pnl_dollars|, so the model
    naturally learns to chase big asymmetric payoffs.

    Anchor regularization prevents catastrophic forgetting of pre-trained
    weights — the model can adapt but not drift arbitrarily.
    """

    def __init__(
        self,
        encoder: MultimodalEncoder,
        contract_decoder: ContractDecoder,
        relevance_scorer: LearnedRelevanceScorer,
        relevance_router: RelevanceRouter,
        lr: float = 1e-4,
        batch_size: int = 32,
        steps_per_cycle: int = 5,
        device: str = "cpu",
        anchor_weight: float = 0.01,
    ):
        self.encoder = encoder
        self.contract_decoder = contract_decoder
        self.relevance_scorer = relevance_scorer
        self.relevance_router = relevance_router
        self.device = device
        self.batch_size = batch_size
        self.steps_per_cycle = steps_per_cycle
        self.anchor_weight = anchor_weight

        # Shared cycle data storage — kept forever
        self._cycle_data: dict[int, SharedCycleData] = {}
        self._next_cycle_id = 0

        # Experience buffer — no cap, grows forever
        self.replay_v2: list[ExperienceV2] = []

        # Train both decoder and relevance scorer
        self.optimizer = torch.optim.Adam(
            list(contract_decoder.parameters())
            + list(relevance_scorer.parameters()),
            lr=lr,
        )
        self.total_steps: int = 0
        self.recent_losses: deque[float] = deque(maxlen=100)

        # Anchor: snapshot pre-trained weights to prevent drift
        self._anchor_params: list[Tensor] = []
        for p in (
            list(contract_decoder.parameters())
            + list(relevance_scorer.parameters())
        ):
            self._anchor_params.append(p.data.clone())

        # Running reward normalization (Welford's online algorithm)
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0  # sum of squared diffs from mean

    def _update_reward_stats(self, reward: float) -> None:
        """Update running mean/variance of rewards for normalization."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_m2 += delta * delta2

    @property
    def _reward_std(self) -> float:
        if self._reward_count < 2:
            return 1.0
        variance = self._reward_m2 / (self._reward_count - 1)
        return max(math.sqrt(variance), 1e-4)

    def record_cycle_data(
        self,
        headline_embs,
        headline_timestamps: list[float],
        headline_texts: list[str],
        macro,
        sentiment: float,
    ) -> int:
        """Store shared cycle data. Returns cycle_id."""
        cycle_id = self._next_cycle_id
        self._next_cycle_id += 1
        self._cycle_data[cycle_id] = SharedCycleData(
            headline_embs=(
                headline_embs
                if isinstance(headline_embs, list)
                else headline_embs.tolist()
            ),
            headline_timestamps=headline_timestamps,
            headline_texts=headline_texts,
            macro=macro if isinstance(macro, list) else macro.tolist(),
            sentiment=sentiment,
            cycle_id=cycle_id,
        )
        return cycle_id

    def record_outcome(
        self,
        cycle_id: int,
        contract_emb,
        contract_text: str,
        target: float,
        direction: str = "BUY",
        pnl_dollars: float = 0.0,
        p_market: float = 0.5,
        liquidity: float = 0.0,
    ) -> None:
        """Record one scored prediction with its P&L and liquidity.

        Zero-pnl samples are still stored for buffer completeness
        but filtered out during learning.
        """
        self.replay_v2.append(
            ExperienceV2(
                cycle_id=cycle_id,
                contract_emb=(
                    contract_emb
                    if isinstance(contract_emb, list)
                    else contract_emb.tolist()
                ),
                contract_text=contract_text,
                target=target,
                direction=direction,
                pnl_dollars=pnl_dollars,
                p_market=p_market,
                liquidity=liquidity,
            )
        )

    def learn_step(self) -> float | None:
        """PnL-weighted policy gradient steps on replay buffer.

        For each experience:
          reward = normalized pnl_dollars
          if BUY:  loss = -reward * log(p_pred)       → win: push p up, lose: push p down
          if SELL: loss = -reward * log(1 - p_pred)   → win: push p down, lose: push p up

        Plus anchor regularization to preserve pre-trained weights.

        Returns avg loss or None if not enough data.
        """
        # Only learn from samples where price actually moved
        valid = [
            e for e in self.replay_v2
            if e.cycle_id in self._cycle_data and e.pnl_dollars != 0.0
        ]
        if len(valid) < self.batch_size:
            return None

        self.contract_decoder.train()
        self.relevance_scorer.train()
        total_loss = 0.0

        for _ in range(self.steps_per_cycle):
            batch = random.sample(valid, min(self.batch_size, len(valid)))

            # Update reward stats for normalization
            for exp in batch:
                self._update_reward_stats(exp.pnl_dollars)

            batch_losses = []
            for exp in batch:
                cycle = self._cycle_data[exp.cycle_id]

                # Build tensors
                h_embs = torch.tensor(
                    cycle.headline_embs, dtype=torch.float32
                ).to(self.device)
                macro_t = torch.tensor(
                    [cycle.macro], dtype=torch.float32
                ).to(self.device)
                sent_t = torch.tensor(
                    [[cycle.sentiment]], dtype=torch.float32
                ).to(self.device)
                contract_emb_t = torch.tensor(
                    exp.contract_emb, dtype=torch.float32
                ).to(self.device)

                # Encode (frozen)
                with torch.no_grad():
                    context_z, headline_tokens = (
                        self.encoder.forward_with_headlines(
                            h_embs.unsqueeze(0), macro_t, sent_t,
                        )
                    )

                # Compute relevance (learned - gradients flow)
                rel_scores = self.relevance_scorer(h_embs, contract_emb_t)

                # Domain scores
                contract_tags = self.relevance_router.domain_graph.tag(
                    exp.contract_text
                )
                domain_scores = torch.zeros(
                    len(cycle.headline_texts), device=self.device
                )
                for i, text in enumerate(cycle.headline_texts):
                    h_tags = self.relevance_router.domain_graph.tag(text)
                    domain_scores[i] = (
                        self.relevance_router.domain_graph.relevance_score(
                            h_tags, contract_tags
                        )
                    )

                alpha = self.relevance_router.alpha
                combined = (
                    alpha * domain_scores + (1 - alpha) * rel_scores
                )
                weights = torch.softmax(combined + 1e-8, dim=0).unsqueeze(
                    0
                )  # (1, N)

                # Predict
                p_pred = self.contract_decoder(
                    context_z,
                    headline_tokens,
                    contract_emb_t.unsqueeze(0),
                    weights,
                ).squeeze()

                # PnL-weighted policy gradient loss
                # Clamp p_pred to avoid log(0)
                p_clamped = torch.clamp(p_pred, 1e-6, 1 - 1e-6)
                if exp.direction == "BUY":
                    log_prob = torch.log(p_clamped)
                else:
                    log_prob = torch.log(1 - p_clamped)

                # Normalize reward by running std for stable gradients
                normalized_reward = exp.pnl_dollars / self._reward_std
                batch_losses.append(-normalized_reward * log_prob)

            policy_loss = torch.stack(batch_losses).mean()

            # Anchor regularization: penalize drift from pre-trained weights
            anchor_loss = torch.tensor(0.0, device=self.device)
            all_params = (
                list(self.contract_decoder.parameters())
                + list(self.relevance_scorer.parameters())
            )
            for p, p0 in zip(all_params, self._anchor_params):
                anchor_loss = anchor_loss + (p - p0.to(self.device)).pow(2).sum()

            loss = policy_loss + self.anchor_weight * anchor_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += policy_loss.item()
            self.total_steps += 1

        self.contract_decoder.eval()
        self.relevance_scorer.eval()
        avg_loss = total_loss / self.steps_per_cycle
        self.recent_losses.append(avg_loss)
        return avg_loss

    def save_weights(self, decoder_path, scorer_path) -> None:
        """Save contract decoder and relevance scorer weights."""
        torch.save(self.contract_decoder.state_dict(), decoder_path)
        torch.save(self.relevance_scorer.state_dict(), scorer_path)
        logger.info(f"Saved contract decoder -> {decoder_path}")
        logger.info(f"Saved relevance scorer -> {scorer_path}")

    def save_buffer(self, path: Path, scorecard=None) -> None:
        """Save replay buffer, cycle data, reward stats, and scorecard."""
        data = {
            "replay_v2": [
                {
                    "cycle_id": e.cycle_id,
                    "contract_emb": e.contract_emb,
                    "contract_text": e.contract_text,
                    "target": e.target,
                    "direction": e.direction,
                    "pnl_dollars": e.pnl_dollars,
                    "p_market": e.p_market,
                    "liquidity": e.liquidity,
                }
                for e in self.replay_v2
            ],
            "cycle_data": {
                str(k): {
                    "headline_embs": v.headline_embs,
                    "headline_timestamps": v.headline_timestamps,
                    "headline_texts": v.headline_texts,
                    "macro": v.macro,
                    "sentiment": v.sentiment,
                    "cycle_id": v.cycle_id,
                }
                for k, v in self._cycle_data.items()
            },
            "next_cycle_id": self._next_cycle_id,
            "total_steps": self.total_steps,
            "anchor_params": [p.cpu() for p in self._anchor_params],
            "reward_stats": {
                "count": self._reward_count,
                "mean": self._reward_mean,
                "m2": self._reward_m2,
            },
        }
        if scorecard is not None:
            data["scorecard"] = scorecard.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        n_nonzero = sum(1 for e in self.replay_v2 if e.pnl_dollars != 0.0)
        logger.info(
            f"Saved replay buffer ({len(self.replay_v2)} experiences, "
            f"{n_nonzero} with P&L signal, "
            f"{len(self._cycle_data)} cycles) -> {path}"
        )

    def load_buffer(self, path: Path) -> dict | None:
        """Load replay buffer + cycle data from disk.

        Returns:
            Scorecard dict if saved, or None.
        """
        if not path.exists():
            return None
        data = torch.load(path, weights_only=False)

        self.replay_v2 = [
            ExperienceV2(**e) for e in data.get("replay_v2", [])
        ]
        self._cycle_data = {
            int(k): SharedCycleData(**v)
            for k, v in data.get("cycle_data", {}).items()
        }
        self._next_cycle_id = data.get("next_cycle_id", 0)
        self.total_steps = data.get("total_steps", 0)

        # Restore anchor params if saved, otherwise keep initial snapshot
        if "anchor_params" in data:
            self._anchor_params = [
                p.to(self.device) for p in data["anchor_params"]
            ]

        # Restore reward normalization stats
        if "reward_stats" in data:
            rs = data["reward_stats"]
            self._reward_count = rs.get("count", 0)
            self._reward_mean = rs.get("mean", 0.0)
            self._reward_m2 = rs.get("m2", 0.0)

        n_nonzero = sum(1 for e in self.replay_v2 if e.pnl_dollars != 0.0)
        logger.info(
            f"Loaded replay buffer ({len(self.replay_v2)} experiences, "
            f"{n_nonzero} with P&L signal, "
            f"{len(self._cycle_data)} cycles) from {path}"
        )
        return data.get("scorecard")

    @property
    def avg_loss(self) -> float | None:
        if not self.recent_losses:
            return None
        return sum(self.recent_losses) / len(self.recent_losses)

    @property
    def buffer_size(self) -> int:
        return len(self.replay_v2)
