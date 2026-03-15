from __future__ import annotations

from pathlib import Path

import torch
import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SeerConfig(BaseSettings):
    # Encoder
    latent_dim: int = 128
    news_embedding_dim: int = 384  # sentence-transformers default
    macro_feature_count: int = 12
    sentiment_dim: int = 1

    # MCTS
    mcts_simulations: int = 1000
    mcts_exploration_constant: float = 1.41
    mcts_rollout_depth: int = 5

    # Alpha detection
    alpha_threshold: float = 0.15
    confidence_threshold: float = 0.8

    # Risk
    max_portfolio_risk: float = 0.02
    kelly_fraction: float = 0.25

    # OOD
    mahalanobis_threshold: float = 3.0

    # Scanner
    scan_interval: int = 300  # seconds between scan cycles (5 min)
    scan_min_edge: float = 0.05  # min edge to surface as opportunity
    scan_mcts_candidates: int = 10  # max contracts to deep-eval with MCTS per cycle
    scan_bet_size: float = 20.0  # simulated bet size in USD per prediction
    scan_min_liquidity: float = 5000.0  # min liquidity in USD to predict (skip dead contracts)

    # Per-contract prediction
    relevance_hidden_dim: int = 256  # LearnedRelevanceScorer hidden layer
    domain_alpha: float = 0.7  # domain graph vs learned scorer balance (1.0 = all domain)
    recency_halflife_hours: float = 6.0  # headline recency exponential decay
    max_headlines_per_cycle: int = 300  # cap headlines for memory
    learning_buffer_size: int = 500000  # replay buffer capacity (~2 days at 5-min cycles)
    learning_max_cycles: int = 2000  # shared headline data retention

    # Hardware
    device: str = "cuda"

    # Paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    synthetic_dir: Path = _PROJECT_ROOT / "data" / "synthetic"
    cache_dir: Path = _PROJECT_ROOT / "data" / "cache"

    model_config = {"env_prefix": "SEER_", "env_file": ".env", "extra": "ignore"}

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return v

    def load_contracts(self) -> dict:
        path = Path(__file__).parent / "contracts.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def load_risk_params(self) -> dict:
        path = Path(__file__).parent / "risk_params.yaml"
        with open(path) as f:
            return yaml.safe_load(f)
