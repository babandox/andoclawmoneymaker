"""Tests for online learning from scored predictions."""

from __future__ import annotations

import torch
import pytest

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.learning import Experience, OnlineLearner, ReplayBuffer
from radiant_seer.planning.reward_module import OutcomeDecoder


# ── ReplayBuffer ─────────────────────────────────────────────────────


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(maxlen=100)
        assert len(buf) == 0
        buf.add(Experience([0.0] * 10, [0.0] * 3, 0.5, 0.6))
        assert len(buf) == 1

    def test_maxlen(self):
        buf = ReplayBuffer(maxlen=5)
        for i in range(10):
            buf.add(Experience([float(i)] * 10, [0.0] * 3, 0.0, 0.5))
        assert len(buf) == 5
        # Oldest should be gone, newest kept
        assert buf.buffer[0].news_emb[0] == 5.0

    def test_sample(self):
        buf = ReplayBuffer(maxlen=100)
        for i in range(20):
            buf.add(Experience([float(i)] * 10, [0.0] * 3, 0.0, 0.5))
        batch = buf.sample(5)
        assert len(batch) == 5
        assert all(isinstance(e, Experience) for e in batch)

    def test_sample_capped(self):
        buf = ReplayBuffer(maxlen=100)
        for i in range(3):
            buf.add(Experience([0.0] * 10, [0.0] * 3, 0.0, 0.5))
        batch = buf.sample(10)
        assert len(batch) == 3


# ── OnlineLearner ────────────────────────────────────────────────────


class TestOnlineLearner:
    def _make_learner(self) -> OnlineLearner:
        config = SeerConfig(device="cpu")
        encoder = MultimodalEncoder(
            news_dim=config.news_embedding_dim,
            macro_dim=config.macro_feature_count,
            latent_dim=config.latent_dim,
        )
        decoder = OutcomeDecoder(latent_dim=config.latent_dim)
        encoder.eval()
        return OnlineLearner(
            encoder=encoder,
            decoder=decoder,
            batch_size=8,
            steps_per_cycle=3,
            device="cpu",
        )

    def test_init(self):
        learner = self._make_learner()
        assert learner.total_steps == 0
        assert learner.buffer_size == 0
        assert learner.avg_loss is None

    def test_record_outcomes(self):
        learner = self._make_learner()
        n = learner.record_outcomes(
            news_emb=[0.0] * 384,
            macro=[0.0] * 12,
            sentiment=0.5,
            outcomes=[0.3, 0.5, 0.7],
        )
        assert n == 3
        assert learner.buffer_size == 3

    def test_learn_step_too_small(self):
        learner = self._make_learner()
        # Buffer too small (< batch_size)
        learner.record_outcomes([0.0] * 384, [0.0] * 12, 0.0, [0.5])
        result = learner.learn_step()
        assert result is None
        assert learner.total_steps == 0

    def test_learn_step_frozen(self):
        """V1 learner is frozen — learn_step returns None without updating."""
        learner = self._make_learner()
        for i in range(20):
            learner.record_outcomes(
                news_emb=[float(i) * 0.01] * 384,
                macro=[float(i) * 0.1] * 12,
                sentiment=0.0,
                outcomes=[0.3 + i * 0.02],
            )
        loss = learner.learn_step()
        assert loss is None
        assert learner.total_steps == 0

    def test_decoder_weights_frozen(self):
        """V1 decoder weights should NOT change — learning is disabled."""
        learner = self._make_learner()
        w_before = learner.decoder.net[0].weight.clone()

        for i in range(20):
            learner.record_outcomes(
                [float(i) * 0.01] * 384,
                [float(i) * 0.1] * 12,
                0.0,
                [0.8],
            )
        learner.learn_step()

        w_after = learner.decoder.net[0].weight
        assert torch.allclose(w_before, w_after), "V1 decoder weights should be frozen"

    def test_buffer_still_records(self):
        """V1 still records outcomes to buffer even though learning is off."""
        learner = self._make_learner()
        learner.record_outcomes([0.1] * 384, [1.0] * 12, 0.0, [0.7] * 10)
        assert learner.buffer_size == 10

    def test_save_decoder(self, tmp_path):
        learner = self._make_learner()
        path = tmp_path / "decoder.pt"
        learner.save_decoder(path)
        assert path.exists()

        # Verify it loads
        new_decoder = OutcomeDecoder(latent_dim=128)
        new_decoder.load_state_dict(torch.load(path, weights_only=True))

    def test_encoder_stays_frozen(self):
        learner = self._make_learner()

        w_before = learner.encoder.news_proj.net[0].weight.clone()

        for i in range(20):
            learner.record_outcomes(
                [float(i) * 0.01] * 384,
                [0.0] * 12,
                0.0,
                [0.5],
            )
        learner.learn_step()

        w_after = learner.encoder.news_proj.net[0].weight
        assert torch.allclose(w_before, w_after), "Encoder weights should NOT change"
