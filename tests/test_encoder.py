"""Tests for the multimodal encoder and VICReg training."""

import pytest
import torch

from radiant_seer.data_swarm.synthetic import SyntheticCivStateGenerator
from radiant_seer.intelligence.loss_functions import VICRegLoss
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder


@pytest.fixture
def encoder():
    return MultimodalEncoder(news_dim=384, macro_dim=12, latent_dim=128)


@pytest.fixture
def dataset():
    gen = SyntheticCivStateGenerator(seed=42)
    return gen.generate_dataset(n_episodes=10, episode_length=20)


class TestMultimodalEncoder:
    def test_output_shape(self, encoder):
        news = torch.randn(4, 384)
        macro = torch.randn(4, 12)
        sentiment = torch.randn(4, 1)

        z = encoder(news, macro, sentiment)
        assert z.shape == (4, 128)

    def test_encode_sequence(self, encoder):
        news = torch.randn(2, 10, 384)
        macro = torch.randn(2, 10, 12)
        sentiment = torch.randn(2, 10, 1)

        z_seq = encoder.encode_sequence(news, macro, sentiment)
        assert z_seq.shape == (2, 10, 128)

    def test_different_inputs_produce_different_outputs(self, encoder):
        news1 = torch.randn(1, 384)
        news2 = torch.randn(1, 384)
        macro = torch.randn(1, 12)
        sentiment = torch.randn(1, 1)

        z1 = encoder(news1, macro, sentiment)
        z2 = encoder(news2, macro, sentiment)
        assert not torch.allclose(z1, z2, atol=1e-4)

    def test_deterministic(self, encoder):
        encoder.eval()
        news = torch.randn(2, 384)
        macro = torch.randn(2, 12)
        sentiment = torch.randn(2, 1)

        z1 = encoder(news, macro, sentiment)
        z2 = encoder(news, macro, sentiment)
        assert torch.allclose(z1, z2)


class TestVICRegLoss:
    def test_loss_components(self):
        loss_fn = VICRegLoss()
        z_a = torch.randn(32, 128)
        z_b = torch.randn(32, 128)

        result = loss_fn(z_a, z_b)
        assert "loss" in result
        assert "invariance" in result
        assert "variance" in result
        assert "covariance" in result
        assert result["loss"].item() > 0

    def test_identical_inputs_low_invariance(self):
        loss_fn = VICRegLoss()
        z = torch.randn(32, 128)

        result = loss_fn(z, z)
        assert result["invariance"].item() < 1e-6

    def test_collapsed_embeddings_high_variance_loss(self):
        loss_fn = VICRegLoss()
        # All embeddings are the same → collapsed → high variance loss
        z = torch.ones(32, 128) * 0.5
        z_b = torch.randn(32, 128)

        result = loss_fn(z, z_b)
        assert result["variance"].item() > 0

    def test_vicreg_training_step(self, encoder, dataset):
        """Verify a single training step runs without error."""
        loss_fn = VICRegLoss()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Get pairs of consecutive states
        news_t = dataset.news[0, :-1]    # (T-1, 384)
        news_t1 = dataset.news[0, 1:]    # (T-1, 384)
        macro_t = dataset.macro[0, :-1]
        macro_t1 = dataset.macro[0, 1:]
        sent_t = dataset.sentiment[0, :-1]
        sent_t1 = dataset.sentiment[0, 1:]

        z_t = encoder(news_t, macro_t, sent_t)
        z_t1 = encoder(news_t1, macro_t1, sent_t1)

        result = loss_fn(z_t, z_t1)
        result["loss"].backward()
        optimizer.step()

        assert result["loss"].item() > 0
        assert not torch.isnan(result["loss"])
