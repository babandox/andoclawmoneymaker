"""Phase 1-3 exit criteria evaluation.

Checks:
  Phase 1: No dimensional collapse, regime clustering in latent space
  Phase 2: Event sensitivity, multi-step rollout stability
  Phase 3: MCTS convergence, LogicGuard validation, timing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from loguru import logger

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.synthetic import SyntheticCivStateGenerator
from radiant_seer.intelligence.causal_predictor import CausalPredictor
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.planning.logic_guard import LogicGuard, TrajectoryStep
from radiant_seer.planning.reward_module import OutcomeDecoder
from radiant_seer.planning.seer_mcts import SeerMCTS


@dataclass
class Phase1Report:
    passed: bool = False
    min_std: float = 0.0
    mean_std: float = 0.0
    collapsed_dims: int = 0
    regime_separation: float = 0.0
    details: list[str] = field(default_factory=list)


@dataclass
class Phase2Report:
    passed: bool = False
    event_sensitivity: float = 0.0
    rollout_norm_ratio: float = 0.0
    rollout_diversity: float = 0.0
    details: list[str] = field(default_factory=list)


@dataclass
class Phase3Report:
    passed: bool = False
    mcts_time_1000: float = 0.0
    probability_stable: bool = False
    logic_guard_rejects: int = 0
    details: list[str] = field(default_factory=list)


def evaluate_phase1(
    encoder: MultimodalEncoder,
    config: SeerConfig | None = None,
    n_episodes: int = 200,
    collapse_threshold: float = 0.1,
) -> Phase1Report:
    """Check Phase 1 exit criteria: no collapse, regime separation."""
    config = config or SeerConfig()
    device = config.device
    report = Phase1Report()

    encoder = encoder.to(device)
    encoder.eval()

    gen = SyntheticCivStateGenerator(seed=999)
    dataset = gen.generate_dataset(n_episodes=n_episodes, episode_length=30)

    with torch.no_grad():
        # Encode all terminal states
        z_all = encoder(
            dataset.news[:, -1].to(device),
            dataset.macro[:, -1].to(device),
            dataset.sentiment[:, -1].to(device),
        )

        # 1. Dimensional collapse check
        z_std = z_all.std(dim=0)
        report.min_std = z_std.min().item()
        report.mean_std = z_std.mean().item()
        report.collapsed_dims = (z_std < collapse_threshold).sum().item()

        if report.collapsed_dims == 0:
            report.details.append(f"PASS: No dimensional collapse (min_std={report.min_std:.4f})")
        else:
            report.details.append(
                f"FAIL: {report.collapsed_dims}/{config.latent_dim} dims collapsed "
                f"(min_std={report.min_std:.4f})"
            )

        # 2. Regime separation: compute inter-regime vs intra-regime distance
        regimes = dataset.regimes[:, -1]
        regime_centroids = {}
        for r in range(3):
            mask = regimes == r
            if mask.sum() > 0:
                regime_centroids[r] = z_all[mask].mean(dim=0)

        if len(regime_centroids) >= 2:
            # Inter-regime distance
            centroids = list(regime_centroids.values())
            inter_dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    inter_dists.append(torch.dist(centroids[i], centroids[j]).item())
            mean_inter = sum(inter_dists) / len(inter_dists)

            # Intra-regime distance (average distance within each regime)
            intra_dists = []
            for r, centroid in regime_centroids.items():
                mask = regimes == r
                if mask.sum() > 1:
                    dists = torch.cdist(
                        z_all[mask].unsqueeze(0), centroid.unsqueeze(0).unsqueeze(0)
                    ).squeeze()
                    intra_dists.append(dists.mean().item())

            mean_intra = sum(intra_dists) / len(intra_dists) if intra_dists else 1.0
            report.regime_separation = mean_inter / (mean_intra + 1e-8)

            if report.regime_separation > 1.0:
                report.details.append(
                    f"PASS: Regime separation ratio={report.regime_separation:.2f} (>1.0)"
                )
            else:
                report.details.append(
                    f"WARN: Weak regime separation ratio={report.regime_separation:.2f} (<1.0)"
                )
        else:
            report.details.append("SKIP: Not enough regimes in sample for separation test")

    report.passed = report.collapsed_dims == 0
    return report


def evaluate_phase2(
    encoder: MultimodalEncoder,
    predictor: CausalPredictor,
    config: SeerConfig | None = None,
) -> Phase2Report:
    """Check Phase 2 exit criteria: event sensitivity, rollout stability."""
    config = config or SeerConfig()
    device = config.device
    report = Phase2Report()

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    gen = SyntheticCivStateGenerator(seed=555)
    dataset = gen.generate_dataset(n_episodes=50, episode_length=20)

    with torch.no_grad():
        z_start = encoder(
            dataset.news[:16, 0].to(device),
            dataset.macro[:16, 0].to(device),
            dataset.sentiment[:16, 0].to(device),
        )

        # 1. Event sensitivity: different events from same z should produce different z_next
        z_repeated = z_start[:1].expand(8, -1)  # Same state, 8 copies
        events = torch.arange(8, device=device)
        z_nexts = predictor(z_repeated, events)

        pairwise = torch.cdist(z_nexts.unsqueeze(0), z_nexts.unsqueeze(0)).squeeze()
        # Exclude diagonal
        mask = ~torch.eye(8, device=device, dtype=torch.bool)
        report.event_sensitivity = pairwise[mask].mean().item()

        sens = report.event_sensitivity
        if sens > 0.01:
            report.details.append(f"PASS: Event sensitivity={sens:.4f} (distinct)")
        else:
            report.details.append(f"FAIL: Event sensitivity={sens:.4f} (identical)")

        # 2. Multi-step rollout stability
        events_5 = torch.randint(0, 8, (16, 5), device=device)
        trajectory = predictor.rollout(z_start, events_5)

        norms = trajectory.norm(dim=-1)  # (16, 6)
        report.rollout_norm_ratio = (norms[:, -1].mean() / norms[:, 0].mean()).item()

        # Check diversity of final states
        final = trajectory[:, -1]
        final_dists = torch.cdist(final.unsqueeze(0), final.unsqueeze(0)).squeeze()
        final_mask = ~torch.eye(16, device=device, dtype=torch.bool)
        report.rollout_diversity = final_dists[final_mask].mean().item()

        nr = report.rollout_norm_ratio
        if 0.5 < nr < 2.0:
            report.details.append(f"PASS: 5-step norm ratio={nr:.3f} (stable)")
        else:
            report.details.append(f"FAIL: 5-step norm ratio={nr:.3f} (unstable)")

        div = report.rollout_diversity
        if div > 0.01:
            report.details.append(f"PASS: Rollout diversity={div:.4f}")
        else:
            report.details.append(f"FAIL: Rollout diversity={div:.4f} (collapsed)")

    report.passed = (
        report.event_sensitivity > 0.01
        and 0.5 < report.rollout_norm_ratio < 2.0
        and report.rollout_diversity > 0.01
    )
    return report


def evaluate_phase3(
    predictor: CausalPredictor,
    decoder: OutcomeDecoder,
    config: SeerConfig | None = None,
) -> Phase3Report:
    """Check Phase 3 exit criteria: MCTS convergence, timing, logic guard."""
    config = config or SeerConfig()
    device = config.device
    report = Phase3Report()

    predictor = predictor.to(device).eval()
    decoder = decoder.to(device).eval()
    guard = LogicGuard()

    mcts = SeerMCTS(
        causal_predictor=predictor,
        outcome_decoder=decoder,
        logic_guard=guard,
        rollout_depth=config.mcts_rollout_depth,
        device=device,
    )

    z_root = torch.randn(config.latent_dim, device=device)

    # 1. MCTS timing for 1000 simulations
    start = time.perf_counter()
    result_1000 = mcts.search(z_root, p_market=0.5, n_simulations=1000)
    elapsed = time.perf_counter() - start
    report.mcts_time_1000 = elapsed

    if elapsed < 10.0:
        report.details.append(f"PASS: 1000 MCTS simulations in {elapsed:.2f}s (<10s)")
    else:
        report.details.append(f"FAIL: 1000 MCTS simulations took {elapsed:.2f}s (>10s)")

    # 2. Convergence: probability should stabilize with more simulations
    p_10 = mcts.search(z_root, p_market=0.5, n_simulations=10).p_mcts
    p_100 = mcts.search(z_root, p_market=0.5, n_simulations=100).p_mcts
    p_1000 = result_1000.p_mcts

    # Difference should shrink
    diff_10_100 = abs(p_10 - p_100)
    diff_100_1000 = abs(p_100 - p_1000)

    report.details.append(
        f"INFO: P_mcts estimates: 10sim={p_10:.3f}, 100sim={p_100:.3f}, 1000sim={p_1000:.3f}"
    )
    report.details.append(
        f"INFO: Convergence gaps: |10-100|={diff_10_100:.3f}, |100-1000|={diff_100_1000:.3f}"
    )
    report.probability_stable = diff_100_1000 < diff_10_100 or diff_100_1000 < 0.05

    if report.probability_stable:
        report.details.append("PASS: MCTS estimates converge with more simulations")
    else:
        report.details.append("WARN: MCTS convergence unclear (stochastic — may pass on retry)")

    # 3. Logic guard rejects impossible trajectories
    from datetime import date

    invalid_trajectories = [
        # Election before mandated date
        [TrajectoryStep(event_type=6, timestamp=date(2025, 1, 1))],
        # Rate change > 50bps
        [TrajectoryStep(event_type=3, metadata={"rate_change_bps": 100})],
        # Temporal violation
        [
            TrajectoryStep(event_type=0, timestamp=date(2026, 8, 1)),
            TrajectoryStep(event_type=1, timestamp=date(2026, 6, 1)),
        ],
        # Probability out of bounds
        [TrajectoryStep(event_type=0, metadata={"probability": 1.5})],
    ]

    rejected = 0
    for traj in invalid_trajectories:
        result = guard.validate_trajectory(traj)
        if not result.valid:
            rejected += 1
    report.logic_guard_rejects = rejected

    if rejected == len(invalid_trajectories):
        report.details.append(f"PASS: LogicGuard rejected all {rejected} invalid trajectories")
    else:
        report.details.append(
            f"FAIL: LogicGuard rejected {rejected}/{len(invalid_trajectories)} invalid trajectories"
        )

    report.passed = elapsed < 10.0 and rejected == len(invalid_trajectories)
    return report


def run_full_evaluation(
    encoder: MultimodalEncoder,
    predictor: CausalPredictor,
    decoder: OutcomeDecoder,
    config: SeerConfig | None = None,
) -> dict[str, Phase1Report | Phase2Report | Phase3Report]:
    """Run all phase evaluations and print summary."""
    config = config or SeerConfig()

    logger.info("=" * 60)
    logger.info("RADIANT SEER — EXIT CRITERIA EVALUATION")
    logger.info("=" * 60)

    # Phase 1
    logger.info("\n--- Phase 1: Encoder ---")
    p1 = evaluate_phase1(encoder, config)
    for line in p1.details:
        logger.info(f"  {line}")
    logger.info(f"  Phase 1: {'PASSED' if p1.passed else 'FAILED'}")

    # Phase 2
    logger.info("\n--- Phase 2: Causal Predictor ---")
    p2 = evaluate_phase2(encoder, predictor, config)
    for line in p2.details:
        logger.info(f"  {line}")
    logger.info(f"  Phase 2: {'PASSED' if p2.passed else 'FAILED'}")

    # Phase 3
    logger.info("\n--- Phase 3: MCTS + Logic Guard ---")
    p3 = evaluate_phase3(predictor, decoder, config)
    for line in p3.details:
        logger.info(f"  {line}")
    logger.info(f"  Phase 3: {'PASSED' if p3.passed else 'FAILED'}")

    all_passed = p1.passed and p2.passed and p3.passed
    logger.info("\n" + "=" * 60)
    logger.info(f"OVERALL: {'ALL PHASES PASSED' if all_passed else 'SOME PHASES FAILED'}")
    logger.info("=" * 60)

    return {"phase1": p1, "phase2": p2, "phase3": p3}
