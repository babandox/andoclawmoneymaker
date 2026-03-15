"""Tests for the causal predictor."""

import pytest
import torch

from radiant_seer.intelligence.causal_predictor import CausalPredictor


@pytest.fixture
def predictor():
    return CausalPredictor(latent_dim=128, num_event_types=8)


class TestCausalPredictor:
    def test_output_shape(self, predictor):
        z = torch.randn(4, 128)
        events = torch.randint(0, 8, (4,))

        z_next = predictor(z, events)
        assert z_next.shape == (4, 128)

    def test_different_events_produce_different_states(self, predictor):
        z = torch.randn(1, 128)
        event_a = torch.tensor([0])
        event_b = torch.tensor([5])

        z_a = predictor(z, event_a)
        z_b = predictor(z, event_b)
        assert not torch.allclose(z_a, z_b, atol=1e-4)

    def test_rollout_shape(self, predictor):
        z_start = torch.randn(2, 128)
        events = torch.randint(0, 8, (2, 5))

        trajectory = predictor.rollout(z_start, events)
        assert trajectory.shape == (2, 6, 128)  # 5 steps + start

    def test_rollout_first_state_matches_start(self, predictor):
        z_start = torch.randn(2, 128)
        events = torch.randint(0, 8, (2, 3))

        trajectory = predictor.rollout(z_start, events)
        assert torch.allclose(trajectory[:, 0], z_start)

    def test_multi_step_stays_in_distribution(self, predictor):
        """Verify 5-step rollouts don't explode or collapse."""
        z_start = torch.randn(8, 128)
        events = torch.randint(0, 8, (8, 5))

        trajectory = predictor.rollout(z_start, events)

        # Check no NaN or Inf
        assert not torch.isnan(trajectory).any()
        assert not torch.isinf(trajectory).any()

        # Check norms stay reasonable (not exploding)
        norms = trajectory.norm(dim=-1)
        assert norms.max() < 100.0, f"Norm explosion: max={norms.max():.1f}"

        # Check not all collapsed to same point
        final_states = trajectory[:, -1]
        pairwise_dists = torch.cdist(final_states.unsqueeze(0), final_states.unsqueeze(0)).squeeze()
        assert pairwise_dists.mean() > 0.01, "States collapsed to same point"

    def test_deterministic_in_eval(self, predictor):
        predictor.eval()
        z = torch.randn(2, 128)
        events = torch.tensor([1, 3])

        z1 = predictor(z, events)
        z2 = predictor(z, events)
        assert torch.allclose(z1, z2)
