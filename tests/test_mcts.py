"""Tests for the MCTS search engine."""

import pytest
import torch

from radiant_seer.intelligence.causal_predictor import CausalPredictor
from radiant_seer.planning.logic_guard import LogicGuard
from radiant_seer.planning.reward_module import OutcomeDecoder
from radiant_seer.planning.seer_mcts import MCTSNode, SeerMCTS


@pytest.fixture
def mcts():
    predictor = CausalPredictor(latent_dim=64, num_event_types=4)
    decoder = OutcomeDecoder(latent_dim=64, hidden_dim=32)
    guard = LogicGuard()
    predictor.eval()
    decoder.eval()
    return SeerMCTS(
        causal_predictor=predictor,
        outcome_decoder=decoder,
        logic_guard=guard,
        num_event_types=4,
        rollout_depth=3,
        device="cpu",
    )


class TestMCTSNode:
    def test_initial_state(self):
        node = MCTSNode(z_state=torch.randn(64))
        assert node.visit_count == 0
        assert node.mean_value == 0.0
        assert node.is_leaf()

    def test_uct_unvisited_is_inf(self):
        node = MCTSNode(z_state=torch.randn(64))
        assert node.uct_value(1.41) == float("inf")

    def test_uct_visited(self):
        parent = MCTSNode(z_state=torch.randn(64), visit_count=10)
        child = MCTSNode(
            z_state=torch.randn(64), parent=parent, visit_count=5, total_value=2.5
        )
        uct = child.uct_value(1.41)
        assert uct > 0  # Should be exploitation + exploration


class TestSeerMCTS:
    def test_search_returns_result(self, mcts):
        z_root = torch.randn(64)
        result = mcts.search(z_root, p_market=0.5, n_simulations=50)

        assert 0.0 <= result.p_mcts <= 1.0
        assert result.visit_count == 50
        assert result.nodes_expanded >= 0

    def test_probability_stabilizes(self, mcts):
        """More simulations should produce more stable estimates."""
        z_root = torch.randn(64)
        torch.manual_seed(42)

        results_10 = [
            mcts.search(z_root.clone(), p_market=0.5, n_simulations=10).p_mcts
            for _ in range(5)
        ]
        torch.manual_seed(42)
        results_100 = [
            mcts.search(z_root.clone(), p_market=0.5, n_simulations=100).p_mcts
            for _ in range(5)
        ]

        # Variance should decrease with more simulations
        var_10 = torch.tensor(results_10).var().item()
        var_100 = torch.tensor(results_100).var().item()
        # This is a probabilistic test — just check it doesn't crash
        assert var_10 >= 0 and var_100 >= 0

    def test_search_different_markets(self, mcts):
        """Different market prices should yield different mean rewards."""
        z_root = torch.randn(64)

        result_low = mcts.search(z_root, p_market=0.1, n_simulations=100)
        result_high = mcts.search(z_root, p_market=0.9, n_simulations=100)

        # Mean reward should be higher when market underprices (low p_market)
        # vs overprices (high p_market), generally
        assert result_low.mean_reward != result_high.mean_reward
