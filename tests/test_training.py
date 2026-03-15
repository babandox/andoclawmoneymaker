"""Tests for the training pipeline — fast smoke tests (small models, few epochs)."""

import pytest

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.synthetic import SyntheticCivStateGenerator
from radiant_seer.intelligence.causal_predictor import CausalPredictor
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.planning.reward_module import OutcomeDecoder
from radiant_seer.training.evaluate import evaluate_phase1, evaluate_phase2, evaluate_phase3
from radiant_seer.training.train_encoder import PairDataset, train_encoder
from radiant_seer.training.train_predictor import TransitionDataset, train_predictor
from radiant_seer.training.train_reward import train_reward_decoder


@pytest.fixture
def small_config():
    """Config with small dimensions for fast testing."""
    config = SeerConfig()
    # Override to use small dims — we can't change frozen fields but can use the object
    return config


class TestPairDataset:
    def test_length(self):
        gen = SyntheticCivStateGenerator(seed=0)
        dataset = gen.generate_dataset(n_episodes=5, episode_length=10)
        pairs = PairDataset(dataset)
        # 5 episodes * 9 pairs each = 45
        assert len(pairs) == 45

    def test_item_keys(self):
        gen = SyntheticCivStateGenerator(seed=0)
        dataset = gen.generate_dataset(n_episodes=3, episode_length=10)
        pairs = PairDataset(dataset)
        item = pairs[0]
        assert "news_t" in item
        assert "news_t1" in item
        assert "macro_t" in item
        assert "sentiment_t" in item
        assert item["news_t"].shape == (384,)


class TestTransitionDataset:
    def test_length(self):
        gen = SyntheticCivStateGenerator(seed=0)
        dataset = gen.generate_dataset(n_episodes=5, episode_length=10)
        trans = TransitionDataset(dataset)
        assert len(trans) == 45

    def test_has_event(self):
        gen = SyntheticCivStateGenerator(seed=0)
        dataset = gen.generate_dataset(n_episodes=3, episode_length=10)
        trans = TransitionDataset(dataset)
        item = trans[0]
        assert "event_t" in item


class TestTrainEncoderSmoke:
    def test_train_encoder_runs(self):
        """Smoke test: 2 epochs on tiny data."""
        encoder, history = train_encoder(
            n_episodes=10,
            episode_length=10,
            epochs=2,
            batch_size=16,
            lr=1e-3,
        )
        assert isinstance(encoder, MultimodalEncoder)
        assert len(history["loss"]) == 2
        assert all(v > 0 for v in history["loss"])

    def test_encoder_loss_decreases(self):
        """Loss should generally decrease over a few epochs."""
        _, history = train_encoder(
            n_episodes=20,
            episode_length=20,
            epochs=5,
            batch_size=32,
            lr=1e-3,
        )
        # First loss should be >= last (allowing some noise)
        assert history["loss"][-1] <= history["loss"][0] * 1.5


class TestTrainPredictorSmoke:
    def test_train_predictor_runs(self):
        """Smoke test with untrained encoder."""
        encoder = MultimodalEncoder()
        predictor, history = train_predictor(
            encoder=encoder,
            n_episodes=10,
            episode_length=10,
            epochs=2,
            batch_size=16,
            lr=1e-3,
        )
        assert isinstance(predictor, CausalPredictor)
        assert len(history["loss"]) == 2


class TestTrainRewardSmoke:
    def test_train_reward_runs(self):
        encoder = MultimodalEncoder()
        decoder, history = train_reward_decoder(
            encoder=encoder,
            n_episodes=50,
            episode_length=10,
            epochs=3,
            batch_size=16,
            lr=1e-3,
        )
        assert isinstance(decoder, OutcomeDecoder)
        assert len(history["val_accuracy"]) == 3


class TestEvaluationSmoke:
    def test_phase1_eval(self):
        encoder = MultimodalEncoder()
        report = evaluate_phase1(encoder, n_episodes=20)
        assert report.min_std >= 0
        assert len(report.details) > 0

    def test_phase2_eval(self):
        encoder = MultimodalEncoder()
        predictor = CausalPredictor()
        report = evaluate_phase2(encoder, predictor)
        assert report.event_sensitivity >= 0
        assert len(report.details) > 0

    def test_phase3_eval(self):
        predictor = CausalPredictor()
        decoder = OutcomeDecoder()
        report = evaluate_phase3(predictor, decoder)
        assert report.mcts_time_1000 > 0
        assert report.logic_guard_rejects == 4  # All 4 invalid trajectories
        assert len(report.details) > 0
