"""Lightweight dashboard — minimal display, maximum performance.

Shows only P&L numbers per model. No charts, no contract lists, no headline
tracking. Designed to run all 4 models on a Mac Mini without UI overhead.

Usage:
    python -m radiant_seer.lightweight_dashboard --decoder-version 1
    python -m radiant_seer.lightweight_dashboard --tag v2 --decoder-version 2
    python -m radiant_seer.lightweight_dashboard --tag v3 --decoder-version 3
    python -m radiant_seer.lightweight_dashboard --tag v4 --decoder-version 4 --interval 60
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.dashboard import (
    DashboardState,
    run_collection_cycle,
    run_predictions,
    save_snapshot,
    count_snapshots,
    _load_models,
    _load_contract_models,
)
from radiant_seer.data_swarm.news_embedder import NewsEmbedder
from radiant_seer.data_swarm.scrapers.fred_scraper import FredScraper
from radiant_seer.data_swarm.scrapers.polymarket_scraper import PolymarketScraper
from radiant_seer.data_swarm.scrapers.reddit_scraper import RedditScraper
from radiant_seer.data_swarm.scrapers.rss_scraper import RssScraper
from radiant_seer.data_swarm.scrapers.truthsocial_scraper import TruthSocialScraper
from radiant_seer.data_swarm.sentiment import HeadlineSentimentAnalyzer
from radiant_seer.execution.kelly_sizing import KellySizer
from radiant_seer.intelligence.contract_decoder import (
    ContractDecoder,
    ContractDecoderV2,
)
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.intelligence.relevance import (
    CausalDomainGraph,
    LearnedRelevanceScorer,
    RelevanceRouter,
)
from radiant_seer.learning import OnlineLearner, OnlineLearnerV2
from radiant_seer.planning.reward_module import OutcomeDecoder
from radiant_seer.scanner import EDGE_BUCKETS, Scorecard

MODEL_DESCRIPTIONS = {
    1: "V1: ContractDecoder, flat $20, predicts from scratch",
    2: "V2: ContractDecoderV2, flat $20, anchored on market price",
    3: "V3: ContractDecoder + Kelly sizing, selective bets",
    4: "V4: Contrarian, no ML, bets against extreme prices",
}


def print_status(state: DashboardState, version: int, cycle_time: float) -> None:
    """Print a compact status line."""
    sc = state.scorecard
    desc = MODEL_DESCRIPTIONS.get(version, f"V{version}")

    # Header
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 70}")
    print(f"  {desc}")
    print(f"  {ts}  |  Cycle #{state.cycle_count}  |  {cycle_time:.1f}s")
    print(f"{'=' * 70}")

    # This cycle
    if state.cycle_scored > 0:
        acc = state.cycle_correct / state.cycle_scored * 100
        pnl_sign = "+" if state.cycle_pnl >= 0 else ""
        print(
            f"  This cycle:  {state.cycle_correct}/{state.cycle_scored} "
            f"correct ({acc:.0f}%)  "
            f"P&L ${pnl_sign}{state.cycle_pnl:,.2f}"
        )
    elif state.cycle_count <= 1:
        print("  This cycle:  first cycle, scoring starts next")

    # Cumulative
    if sc.total_scored > 0:
        pnl_sign = "+" if sc.total_pnl_dollars >= 0 else ""
        print(
            f"  Cumulative:  {sc.total_correct}/{sc.total_scored} "
            f"({sc.accuracy:.1%})  "
            f"P&L ${pnl_sign}{sc.total_pnl_dollars:,.2f}"
        )

    # Contracts this cycle
    n_pending = len(state.pending_predictions)
    print(f"  Contracts:   {state.poly_count} available, {n_pending} betting on")

    # Learning (v1-v3 only)
    if state.learning_active and state.learning_total_steps > 0:
        loss_str = f"loss {state.learning_loss:.4f}" if state.learning_loss is not None else ""
        print(
            f"  Learning:    buffer {state.learning_buffer_size}  |  "
            f"{state.learning_total_steps} steps  |  {loss_str}"
        )

    print(f"{'─' * 70}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Radiant Seer — Lightweight")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Cycle interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for save files (e.g., 'v2', 'v3', 'v4')",
    )
    parser.add_argument(
        "--decoder-version", type=int, default=1, choices=[1, 2, 3, 4],
        help="1=from scratch, 2=market-anchored, 3=Kelly, 4=contrarian",
    )
    args = parser.parse_args()

    config = SeerConfig()
    bet_size = config.scan_bet_size
    tag = f"_{args.tag}" if args.tag else ""
    scan_log_path = config.data_dir / f"scan_log{tag}.jsonl"

    output_dir = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Kelly / contrarian setup
    kelly_sizer: KellySizer | None = None
    kelly_bankroll = 10000.0
    contrarian_mode = False
    if args.decoder_version in (3, 4):
        kelly_sizer = KellySizer(
            kelly_fraction=config.kelly_fraction,
            max_risk=config.max_portfolio_risk,
            min_edge=config.scan_min_edge,
        )
    if args.decoder_version == 4:
        contrarian_mode = True

    # Scrapers
    embedder = NewsEmbedder()
    scrapers = {
        "rss": RssScraper(),
        "reddit": RedditScraper(),
        "truthsocial": TruthSocialScraper(),
        "fred": FredScraper(api_key=os.environ.get("FRED_API_KEY", "")),
        "polymarket": PolymarketScraper(apply_filter=False),
        "embedder": embedder,
        "sentiment_analyzer": HeadlineSentimentAnalyzer(embedder),
    }

    # State
    state = DashboardState()
    state.snapshots_on_disk = count_snapshots(output_dir)

    if contrarian_mode:
        state.predictions_active = True
        state.learning_active = False

    # Load models (not needed for v4)
    encoder: MultimodalEncoder | None = None
    decoder: OutcomeDecoder | None = None
    learner: OnlineLearner | None = None
    contract_decoder: ContractDecoder | ContractDecoderV2 | None = None
    relevance_router: RelevanceRouter | None = None
    learner_v2: OnlineLearnerV2 | None = None
    contract_emb_cache: dict[str, np.ndarray] = {}
    models_dir = config.project_root / "data" / "models"
    cd_save_path = models_dir / f"contract_decoder{tag}.pt"
    rs_save_path = models_dir / f"relevance_scorer{tag}.pt"
    buffer_save_path = models_dir / f"replay_buffer{tag}.pt"

    models = _load_models(config)
    if models is not None:
        encoder, decoder = models
        state.predictions_active = True
        learner = OnlineLearner(
            encoder=encoder, decoder=decoder, device=config.device,
        )
        state.learning_active = True

        domain_graph = CausalDomainGraph()
        relevance_scorer = LearnedRelevanceScorer(
            emb_dim=config.news_embedding_dim,
        ).to(config.device)

        use_v2 = args.decoder_version == 2
        if use_v2:
            contract_decoder = ContractDecoderV2(
                latent_dim=config.latent_dim,
                contract_emb_dim=config.news_embedding_dim,
            ).to(config.device)
        else:
            contract_decoder = ContractDecoder(
                latent_dim=config.latent_dim,
                contract_emb_dim=config.news_embedding_dim,
            ).to(config.device)

        contract_models = _load_contract_models(
            config, cd_path=cd_save_path, rs_path=rs_save_path,
            use_v2=use_v2,
        )
        if contract_models is not None:
            contract_decoder, relevance_scorer = contract_models

        relevance_router = RelevanceRouter(
            domain_graph=domain_graph,
            learned_scorer=relevance_scorer,
        )
        learner_v2 = OnlineLearnerV2(
            encoder=encoder,
            contract_decoder=contract_decoder,
            relevance_scorer=relevance_scorer,
            relevance_router=relevance_router,
            device=config.device,
        )
        saved_scorecard = learner_v2.load_buffer(buffer_save_path)
        if saved_scorecard is not None:
            state.scorecard = Scorecard.from_dict(saved_scorecard)

    # Restore scorecard from scan log
    if scan_log_path.exists():
        state.scorecard = Scorecard.from_scan_log(scan_log_path)

    desc = MODEL_DESCRIPTIONS.get(args.decoder_version, "")
    logger.info(f"Starting {desc}")
    logger.info(f"Interval: {args.interval}s  |  Log: {scan_log_path}")

    # Periodic checkpoint
    _CHECKPOINT_INTERVAL = 30 * 60
    _last_checkpoint = time.monotonic()

    try:
        while True:
            state.cycle_count += 1
            t0 = time.monotonic()

            try:
                snapshot = run_collection_cycle(
                    state, scrapers, prices_only=contrarian_mode,
                )

                if not contrarian_mode and state.headlines and embedder is not None:
                    h_embs, h_ts = embedder.embed_with_timestamps(state.headlines)
                    snapshot["headline_embeddings"] = h_embs.tolist()
                    snapshot["headline_timestamps"] = h_ts
                    snapshot["headline_texts"] = state.headlines

                if encoder is not None and decoder is not None or contrarian_mode:
                    run_predictions(
                        state, snapshot, encoder, decoder,
                        config.device, bet_size, scan_log_path,
                        learner=learner,
                        contract_decoder=contract_decoder,
                        relevance_router=relevance_router,
                        learner_v2=learner_v2,
                        embedder=embedder,
                        contract_emb_cache=contract_emb_cache,
                        kelly_sizer=kelly_sizer,
                        kelly_bankroll=kelly_bankroll,
                        contrarian_mode=contrarian_mode,
                    )
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            elapsed = time.monotonic() - t0

            save_snapshot(snapshot, output_dir)

            # Periodic checkpoint
            if time.monotonic() - _last_checkpoint >= _CHECKPOINT_INTERVAL:
                try:
                    if learner is not None and learner.total_steps > 0:
                        decoder_save_path = models_dir / "outcome_decoder.pt"
                        learner.save_decoder(decoder_save_path)
                    if learner_v2 is not None:
                        if learner_v2.total_steps > 0:
                            learner_v2.save_weights(cd_save_path, rs_save_path)
                        if learner_v2.buffer_size > 0:
                            learner_v2.save_buffer(
                                buffer_save_path, scorecard=state.scorecard
                            )
                except Exception as e:
                    logger.error(f"Checkpoint error: {e}")
                _last_checkpoint = time.monotonic()

            # Print status
            print_status(state, args.decoder_version, elapsed)

            # Sleep
            sleep_time = max(0, args.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        sc = state.scorecard
        print(f"\nStopped after {state.cycle_count} cycles.")
        if sc.total_scored > 0:
            print(
                f"Final P&L: ${sc.total_pnl_dollars:+,.2f}  "
                f"({sc.total_correct}/{sc.total_scored} correct, {sc.accuracy:.1%})"
            )


if __name__ == "__main__":
    main()
