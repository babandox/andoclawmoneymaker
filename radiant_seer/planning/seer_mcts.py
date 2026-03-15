"""MCTS Search Engine — "The Psychohistory Tree"

Searches the tree of possible futures to estimate P_mcts for contract outcomes.
Uses UCT selection, CausalPredictor rollouts, and LogicGuard pruning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from radiant_seer.intelligence.causal_predictor import CausalPredictor
from radiant_seer.planning.logic_guard import LogicGuard, TrajectoryStep
from radiant_seer.planning.reward_module import OutcomeDecoder


@dataclass
class MCTSResult:
    p_mcts: float
    visit_count: int
    mean_reward: float
    tree_depth: int
    nodes_expanded: int


@dataclass
class MCTSNode:
    z_state: Tensor
    event: int | None = None
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    depth: int = 0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def uct_value(self, exploration_constant: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploitation = self.mean_value
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / self.visit_count
        )
        return exploitation + exploration


class SeerMCTS:
    """Monte Carlo Tree Search over latent future states."""

    def __init__(
        self,
        causal_predictor: CausalPredictor,
        outcome_decoder: OutcomeDecoder,
        logic_guard: LogicGuard | None = None,
        num_event_types: int = 8,
        exploration_constant: float = 1.41,
        rollout_depth: int = 5,
        device: str = "cpu",
    ):
        self.predictor = causal_predictor
        self.decoder = outcome_decoder
        self.guard = logic_guard or LogicGuard()
        self.num_event_types = num_event_types
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.device = device

    @torch.no_grad()
    def search(
        self,
        z_root: Tensor,
        p_market: float,
        n_simulations: int = 1000,
    ) -> MCTSResult:
        """Run MCTS search from root state.

        Args:
            z_root: (latent_dim,) initial latent state.
            p_market: Current market price for reward computation.
            n_simulations: Number of MCTS iterations.

        Returns:
            MCTSResult with estimated probability and search statistics.
        """
        root = MCTSNode(z_state=z_root.to(self.device), depth=0)
        nodes_expanded = 0
        max_depth = 0

        for _ in range(n_simulations):
            # 1. Selection — traverse tree using UCT
            node = self._select(root)

            # 2. Expansion — add children if not at max depth
            if node.depth < self.rollout_depth and node.visit_count > 0:
                node, expanded = self._expand(node)
                nodes_expanded += expanded

            # 3. Simulation — rollout to terminal state
            reward = self._simulate(node, p_market)

            # 4. Backpropagation
            self._backpropagate(node, reward)
            max_depth = max(max_depth, node.depth)

        # Estimate P_mcts from root's children visit distribution
        p_mcts = self._estimate_probability(root)

        return MCTSResult(
            p_mcts=p_mcts,
            visit_count=root.visit_count,
            mean_reward=root.mean_value,
            tree_depth=max_depth,
            nodes_expanded=nodes_expanded,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node via UCT."""
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.uct_value(self.c))
        return node

    def _expand(self, node: MCTSNode) -> tuple[MCTSNode, int]:
        """Expand node by adding children for each valid event type."""
        expanded = 0
        for event_type in range(self.num_event_types):
            # Check logic guard for valid transition
            trajectory = [
                TrajectoryStep(event_type=node.event or 0),
                TrajectoryStep(event_type=event_type),
            ]
            validation = self.guard.validate_trajectory(trajectory)
            if not validation.valid:
                continue

            # Predict next state
            z_t = node.z_state.unsqueeze(0)
            event = torch.tensor([event_type], device=self.device)
            z_next = self.predictor(z_t, event).squeeze(0)

            child = MCTSNode(
                z_state=z_next,
                event=event_type,
                parent=node,
                depth=node.depth + 1,
            )
            node.children.append(child)
            expanded += 1

        # Return a random child if expansion happened, else node itself
        if node.children:
            idx = torch.randint(len(node.children), (1,)).item()
            return node.children[idx], expanded
        return node, 0

    def _simulate(self, node: MCTSNode, p_market: float) -> float:
        """Rollout from node to estimate reward."""
        z = node.z_state.unsqueeze(0)
        remaining_depth = self.rollout_depth - node.depth

        # Roll forward with random events
        for _ in range(remaining_depth):
            event = torch.randint(self.num_event_types, (1,), device=self.device)
            z = self.predictor(z, event)

        # Decode terminal state to probability
        p_model = self.decoder(z).item()
        reward = p_model - p_market

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def _estimate_probability(self, root: MCTSNode) -> float:
        """Estimate contract probability from MCTS results.

        Uses the outcome decoder on the most-visited child's state,
        weighted by visit distribution.
        """
        if not root.children:
            return self.decoder(root.z_state.unsqueeze(0)).item()

        # Weighted average of decoded probabilities by visit count
        total_visits = sum(c.visit_count for c in root.children)
        if total_visits == 0:
            return 0.5

        p_mcts = 0.0
        for child in root.children:
            if child.visit_count > 0:
                p_child = self.decoder(child.z_state.unsqueeze(0)).item()
                weight = child.visit_count / total_visits
                p_mcts += weight * p_child

        return p_mcts
