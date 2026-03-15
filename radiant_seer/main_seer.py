"""Radiant Seer — Main orchestrator.

Unified loop:
  1. Scrape latest data from ALL sources (RSS, FRED, Polymarket — all markets)
  2. Normalize → state tensor
  3. Encode → latent z
  4. OOD check → emergency skip if out-of-distribution
  5. Mass scan: fast-predict ALL contracts, score previous predictions
  6. Deep eval: MCTS on top opportunities from the scan
  7. Execute trades on validated alpha
  8. Log everything — predictions, scores, trades
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.ingest import DataIngestor
from radiant_seer.data_swarm.news_embedder import NewsEmbedder
from radiant_seer.data_swarm.normalization import StateNormalizer
from radiant_seer.execution.kelly_sizing import KellySizer
from radiant_seer.execution.poly_interface import OrderSide, PolymarketInterface
from radiant_seer.intelligence.causal_predictor import CausalPredictor
from radiant_seer.intelligence.contract_decoder import ContractDecoder
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.intelligence.relevance import (
    CausalDomainGraph,
    LearnedRelevanceScorer,
    RelevanceRouter,
)
from radiant_seer.planning.logic_guard import LogicGuard
from radiant_seer.planning.reward_module import OutcomeDecoder, RewardModule
from radiant_seer.planning.seer_mcts import SeerMCTS
from radiant_seer.scanner import EDGE_BUCKETS, Prediction, ScanResult, Scorecard


@dataclass
class AlphaOpportunity:
    contract_id: str
    question: str
    p_mcts: float
    p_market: float
    edge: float
    confidence: float
    position_fraction: float
    direction: str


class OODDetector:
    """Out-of-distribution detection via Mahalanobis distance."""

    def __init__(self, threshold: float = 3.0, min_samples: int = 50):
        self.threshold = threshold
        self.min_samples = min_samples
        self._mean: Tensor | None = None
        self._inv_cov: Tensor | None = None
        self._n_samples: int = 0
        self._fitted = False

    def fit(self, z_train: Tensor) -> None:
        """Fit distribution from training latent states."""
        self._n_samples = len(z_train)
        self._mean = z_train.mean(dim=0)
        centered = z_train - self._mean
        cov = (centered.T @ centered) / (len(z_train) - 1)
        # Regularize proportional to sample deficit
        reg = max(1e-4, 1.0 / self._n_samples)
        cov += torch.eye(cov.shape[0], device=cov.device) * reg
        self._inv_cov = torch.linalg.inv(cov)
        self._fitted = True

    def load(self, path: Path) -> None:
        """Load pre-fitted OOD baseline from disk."""
        data = torch.load(path, weights_only=True)
        self._mean = data["mean"]
        self._inv_cov = data["inv_cov"]
        self._n_samples = data.get("n_samples", 0)
        self._fitted = True

    def mahalanobis_distance(self, z: Tensor) -> float:
        if not self._fitted:
            return 0.0
        diff = z - self._mean.to(z.device)
        inv_cov = self._inv_cov.to(z.device)
        dist_sq = diff @ inv_cov @ diff
        return dist_sq.sqrt().item()

    def is_ood(self, z: Tensor) -> bool:
        # Don't trust OOD with too few samples — skip the check
        if self._n_samples < self.min_samples:
            return False
        return self.mahalanobis_distance(z) > self.threshold


class RadiantSeer:
    """Main orchestrator for the Radiant Seer predictive engine.

    Runs a unified loop that mass-scans all markets, scores predictions
    against price movements, deep-evaluates top opportunities with MCTS,
    and executes trades on validated alpha.
    """

    def __init__(self, config: SeerConfig | None = None):
        self.config = config or SeerConfig()
        self.device = self.config.device
        logger.info(f"Initializing Radiant Seer on device: {self.device}")

        # Intelligence core
        self.encoder = MultimodalEncoder(
            news_dim=self.config.news_embedding_dim,
            macro_dim=self.config.macro_feature_count,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.predictor = CausalPredictor(
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.outcome_decoder = OutcomeDecoder(
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        # Planning
        self.logic_guard = LogicGuard()
        self.reward_module = RewardModule(self.outcome_decoder)
        self.mcts = SeerMCTS(
            causal_predictor=self.predictor,
            outcome_decoder=self.outcome_decoder,
            logic_guard=self.logic_guard,
            exploration_constant=self.config.mcts_exploration_constant,
            rollout_depth=self.config.mcts_rollout_depth,
            device=self.device,
        )

        # Data
        self.normalizer = StateNormalizer(
            macro_dim=self.config.macro_feature_count,
            news_dim=self.config.news_embedding_dim,
        )
        self.ingestor: DataIngestor | None = None

        # Execution
        self.kelly = KellySizer(
            kelly_fraction=self.config.kelly_fraction,
            max_risk=self.config.max_portfolio_risk,
            min_edge=self.config.alpha_threshold,
        )
        self.exchange = PolymarketInterface(paper_mode=True)

        # Per-contract prediction models
        self.contract_decoder = ContractDecoder(
            latent_dim=self.config.latent_dim,
            contract_emb_dim=self.config.news_embedding_dim,
        ).to(self.device)

        self.relevance_scorer = LearnedRelevanceScorer(
            emb_dim=self.config.news_embedding_dim,
        ).to(self.device)

        self.domain_graph = CausalDomainGraph()
        self.relevance_router = RelevanceRouter(
            domain_graph=self.domain_graph,
            learned_scorer=self.relevance_scorer,
        )

        # News embedder for contract question embedding
        self._news_embedder: NewsEmbedder | None = None
        self._contract_emb_cache: dict[str, np.ndarray] = {}

        # OOD protection
        self.ood_detector = OODDetector(
            threshold=self.config.mahalanobis_threshold
        )

        # Scanner state — restore from scan log if it exists
        self._scan_log_path = self.config.data_dir / "scan_log.jsonl"
        self._scorecard = Scorecard.from_scan_log(self._scan_log_path) if self._scan_log_path.exists() else Scorecard()
        self._pending = self._restore_pending()


        # State
        self._questions: dict[str, str] = {}
        self._cycle_count = 0
        self._last_headline_embs: np.ndarray | None = None
        self._last_headline_timestamps: list[float] = []
        self._last_headline_texts: list[str] = []

        # Set models to eval mode
        self.encoder.eval()
        self.predictor.eval()
        self.outcome_decoder.eval()
        self.contract_decoder.eval()
        self.relevance_scorer.eval()

    def load_models(self, model_dir: Path | None = None) -> None:
        """Load trained model weights and OOD baseline from disk."""
        model_dir = model_dir or (self.config.project_root / "data" / "models")

        predictor_path = model_dir / "predictor.pt"
        decoder_path = model_dir / "outcome_decoder.pt"

        # Prefer fine-tuned encoder + real OOD baseline if available
        ft_encoder = model_dir / "encoder_finetuned.pt"
        base_encoder = model_dir / "encoder_vicreg.pt"
        real_ood = model_dir / "ood_baseline_real.pt"
        base_ood = model_dir / "ood_baseline.pt"

        if ft_encoder.exists():
            self.encoder.load_state_dict(
                torch.load(ft_encoder, weights_only=True)
            )
            logger.info("Loaded fine-tuned encoder")
        elif base_encoder.exists():
            self.encoder.load_state_dict(
                torch.load(base_encoder, weights_only=True)
            )
            logger.info("Loaded base encoder")

        if predictor_path.exists():
            self.predictor.load_state_dict(
                torch.load(predictor_path, weights_only=True)
            )
            logger.info("Loaded predictor weights")

        if decoder_path.exists():
            self.outcome_decoder.load_state_dict(
                torch.load(decoder_path, weights_only=True)
            )
            logger.info("Loaded outcome decoder weights")

        if real_ood.exists():
            self.ood_detector.load(real_ood)
            logger.info("Loaded real-data OOD baseline")
        elif base_ood.exists():
            self.ood_detector.load(base_ood)
            logger.info("Loaded synthetic OOD baseline")

        # Per-contract model weights (optional)
        cd_path = model_dir / "contract_decoder.pt"
        if cd_path.exists():
            self.contract_decoder.load_state_dict(
                torch.load(cd_path, weights_only=True)
            )
            logger.info("Loaded contract decoder weights")

        rs_path = model_dir / "relevance_scorer.pt"
        if rs_path.exists():
            self.relevance_scorer.load_state_dict(
                torch.load(rs_path, weights_only=True)
            )
            logger.info("Loaded relevance scorer weights")

        self.encoder.eval()
        self.predictor.eval()
        self.outcome_decoder.eval()
        self.contract_decoder.eval()
        self.relevance_scorer.eval()

    def init_data_pipeline(
        self,
        fred_api_key: str | None = None,
        contract_ids: list[str] | None = None,
        fetch_all_markets: bool = True,
    ) -> None:
        """Initialize the live data ingestion pipeline.

        Args:
            fred_api_key: FRED API key for macro data.
            contract_ids: Specific contract IDs to always include.
            fetch_all_markets: If True, fetch ALL active Polymarket markets
                (not just tagged ones). Default True for mass scanning.
        """
        from radiant_seer.data_swarm.scrapers.polymarket_scraper import (
            PolymarketScraper,
        )

        self.ingestor = DataIngestor(
            normalizer=self.normalizer,
            fred_api_key=fred_api_key or os.environ.get("FRED_API_KEY"),
            contract_ids=contract_ids or [],
            news_dim=self.config.news_embedding_dim,
        )

        if fetch_all_markets:
            self.ingestor.polymarket = PolymarketScraper(
                contract_ids=contract_ids or [],
                apply_filter=False,
            )
            logger.info("Data pipeline initialized (unfiltered within tags)")
        else:
            logger.info("Data pipeline initialized (filtered markets)")

    @torch.no_grad()
    def encode_state(
        self, news: Tensor, macro: Tensor, sentiment: Tensor
    ) -> Tensor:
        """Encode raw state inputs to latent vector."""
        return self.encoder(
            news.unsqueeze(0).to(self.device),
            macro.unsqueeze(0).to(self.device),
            sentiment.unsqueeze(0).to(self.device),
        ).squeeze(0)

    @torch.no_grad()
    def _fast_predict(self, z: Tensor) -> float:
        """Fast prediction: latent → decoder → probability. No MCTS."""
        return self.outcome_decoder(z.unsqueeze(0)).item()

    @torch.no_grad()
    def evaluate_contract(
        self, z: Tensor, contract_id: str, p_market: float
    ) -> AlphaOpportunity | None:
        """Deep-evaluate a single contract with MCTS."""
        if self.ood_detector.is_ood(z):
            logger.warning(f"OOD detected for {contract_id} — skipping")
            return None

        result = self.mcts.search(
            z_root=z,
            p_market=p_market,
            n_simulations=self.config.mcts_simulations,
        )

        edge = result.p_mcts - p_market
        abs_edge = abs(edge)
        confidence = min(
            1.0, result.visit_count / self.config.mcts_simulations
        )

        question = self._questions.get(contract_id, contract_id[:20])
        logger.info(
            f"[{question[:50]}] P_mcts={result.p_mcts:.3f} "
            f"P_market={p_market:.3f} edge={edge:+.3f} conf={confidence:.2f}"
        )

        if abs_edge < self.config.alpha_threshold:
            return None
        if confidence < self.config.confidence_threshold:
            return None

        position = self.kelly.compute(
            p_model=result.p_mcts,
            p_market=p_market,
            confidence=confidence,
        )

        if position.fraction <= 0:
            return None

        return AlphaOpportunity(
            contract_id=contract_id,
            question=question,
            p_mcts=result.p_mcts,
            p_market=p_market,
            edge=edge,
            confidence=confidence,
            position_fraction=position.fraction,
            direction=position.direction,
        )

    def _get_contract_emb(self, question: str) -> np.ndarray:
        """Get or compute and cache a contract question embedding."""
        if question in self._contract_emb_cache:
            return self._contract_emb_cache[question]
        if self._news_embedder is None:
            self._news_embedder = NewsEmbedder(dim=self.config.news_embedding_dim)
        emb = self._news_embedder.embed([question])[0]
        self._contract_emb_cache[question] = emb
        return emb

    def _scan_and_score(
        self,
        z: Tensor,
        contracts: dict[str, float],
    ) -> ScanResult:
        """Mass scan: fast-predict all contracts, score previous predictions.

        Uses per-contract predictions when headline-level data is available,
        otherwise falls back to single-p_model approach.

        Returns:
            ScanResult with accuracy stats and top opportunities.
        """
        ts = datetime.now().isoformat()

        # Determine if we can do per-contract predictions
        use_per_contract = (
            self._last_headline_embs is not None
            and len(self._last_headline_texts) > 0
        )

        if use_per_contract:
            h_embs_t = torch.tensor(
                self._last_headline_embs, dtype=torch.float32
            ).to(self.device)
            macro_t = z.new_zeros(1, self.config.macro_feature_count)  # placeholder
            sent_t = z.new_zeros(1, 1)

            with torch.no_grad():
                context_z, headline_tokens = self.encoder.forward_with_headlines(
                    h_embs_t.unsqueeze(0), macro_t, sent_t,
                )
        else:
            p_model = self._fast_predict(z)

        # 1. Score previous cycle's predictions against current prices
        bet = self.config.scan_bet_size
        scored: list[Prediction] = []
        for cid, pred in self._pending.items():
            if cid not in contracts:
                continue
            p_now = contracts[cid]
            price_move = p_now - pred.p_market
            direction_sign = 1.0 if pred.direction == "BUY" else -1.0
            profit = price_move * direction_sign

            # Dollar P&L: $20 bet on shares at entry price
            # BUY YES at p_market → shares = bet/p_market → pnl = shares * (p_after - p_market)
            # SELL (BUY NO) at (1-p_market) → pnl = shares * (p_market - p_after)
            if pred.direction == "BUY":
                entry_price = pred.p_market
            else:
                entry_price = 1.0 - pred.p_market
            pnl_dollars = bet * price_move * direction_sign / entry_price if entry_price > 0 else 0.0

            pred.scored = True
            pred.score_timestamp = datetime.now().isoformat()
            pred.p_market_after = round(p_now, 6)
            pred.price_move = round(price_move, 6)
            pred.correct = profit > 0
            pred.profit_pct = round(profit, 6)
            pred.pnl_dollars = round(pnl_dollars, 2)

            self._scorecard.record_score(pred)
            self._append_scan_log({
                "type": "score",
                "prediction_timestamp": pred.timestamp,
                "score_timestamp": pred.score_timestamp,
                "contract_id": pred.contract_id,
                "question": pred.question,
                "p_model": pred.p_model,
                "p_market": pred.p_market,
                "p_market_after": pred.p_market_after,
                "edge": pred.edge,
                "price_move": pred.price_move,
                "direction": pred.direction,
                "correct": pred.correct,
                "profit_pct": pred.profit_pct,
                "pnl_dollars": pred.pnl_dollars,
                "bet_size": bet,
            })
            scored.append(pred)

        # 2. Predict all contracts
        new_pending: dict[str, Prediction] = {}
        opportunities: list[Prediction] = []

        # Pre-compute headline tags ONCE
        cached_headline_tags = None
        if use_per_contract:
            cached_headline_tags = self.relevance_router.cache_headline_tags(
                self._last_headline_texts
            )

        for cid, p_market in contracts.items():
            if p_market <= 0.01 or p_market >= 0.99:
                continue

            if use_per_contract:
                question = self._questions.get(cid, "")
                c_emb_np = self._get_contract_emb(question)
                c_emb_t = torch.tensor(
                    c_emb_np, dtype=torch.float32
                ).to(self.device)

                weights = self.relevance_router.compute_weights(
                    headline_embs=h_embs_t,
                    headline_timestamps=self._last_headline_timestamps,
                    contract_emb=c_emb_t,
                    headline_texts=self._last_headline_texts,
                    contract_text=question,
                    headline_tags=cached_headline_tags,
                )

                with torch.no_grad():
                    p_model_contract = self.contract_decoder(
                        context_z,
                        headline_tokens,
                        c_emb_t.unsqueeze(0),
                        weights.unsqueeze(0),
                    ).item()

                edge = p_model_contract - p_market
                pred = Prediction(
                    timestamp=ts,
                    contract_id=cid,
                    question=question,
                    p_model=round(p_model_contract, 6),
                    p_market=round(p_market, 6),
                    edge=round(edge, 6),
                    direction="BUY" if edge > 0 else "SELL",
                )
            else:
                edge = p_model - p_market
                pred = Prediction(
                    timestamp=ts,
                    contract_id=cid,
                    question=self._questions.get(cid, ""),
                    p_model=round(p_model, 6),
                    p_market=round(p_market, 6),
                    edge=round(edge, 6),
                    direction="BUY" if edge > 0 else "SELL",
                )

            new_pending[cid] = pred
            self._scorecard.record_prediction()
            self._append_scan_log({
                "type": "prediction",
                "timestamp": pred.timestamp,
                "contract_id": pred.contract_id,
                "question": pred.question,
                "p_model": pred.p_model,
                "p_market": pred.p_market,
                "edge": pred.edge,
                "direction": pred.direction,
            })

            if abs(edge) >= self.config.scan_min_edge:
                opportunities.append(pred)

        self._pending = new_pending

        n_correct = sum(1 for s in scored if s.correct)
        accuracy = n_correct / len(scored) if scored else 0.0
        cycle_pnl = sum(s.pnl_dollars or 0.0 for s in scored)

        if scored:
            logger.info(
                f"SCAN: {len(new_pending)} predictions, "
                f"{len(scored)} scored ({n_correct} correct = {accuracy:.1%}), "
                f"cycle P&L ${cycle_pnl:+,.2f} | "
                f"cumulative ${self._scorecard.total_pnl_dollars:+,.2f}"
            )
        else:
            logger.info(
                f"SCAN: {len(new_pending)} predictions (first cycle, nothing to score yet)"
            )

        return ScanResult(
            timestamp=ts,
            n_contracts=len(contracts),
            n_predictions=len(new_pending),
            n_scored=len(scored),
            n_correct=n_correct,
            accuracy=accuracy,
            cycle_pnl=cycle_pnl,
            opportunities=sorted(
                opportunities, key=lambda p: abs(p.edge), reverse=True
            ),
            scored_predictions=scored,
        )

    def _restore_pending(self) -> dict[str, Prediction]:
        """Restore the last batch of unscored predictions from the scan log."""
        if not self._scan_log_path.exists():
            return {}

        last_ts = None
        pending: dict[str, Prediction] = {}

        with open(self._scan_log_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "prediction":
                    ts = rec["timestamp"]
                    if last_ts is None or ts != last_ts:
                        if ts > (last_ts or ""):
                            last_ts = ts
                            pending = {}
                    if ts == last_ts:
                        pending[rec["contract_id"]] = Prediction(
                            timestamp=rec["timestamp"],
                            contract_id=rec["contract_id"],
                            question=rec["question"],
                            p_model=rec["p_model"],
                            p_market=rec["p_market"],
                            edge=rec["edge"],
                            direction=rec["direction"],
                        )

        if pending:
            logger.info(f"Restored {len(pending)} pending predictions from last cycle")
        return pending

    def _append_scan_log(self, record: dict) -> None:
        """Append a record to the JSONL scan log."""
        self._scan_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._scan_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run_cycle(
        self,
        state: dict[str, Tensor] | None = None,
        contracts: dict[str, float] | None = None,
    ) -> list[AlphaOpportunity]:
        """Run one full cycle: ingest → scan all → deep eval top → trade.

        If state/contracts not provided, ingests live data automatically.
        """
        self._cycle_count += 1

        # 1. Ingest live data if not provided
        if state is None or contracts is None:
            if self.ingestor is None:
                raise RuntimeError(
                    "No state provided and data pipeline not initialized. "
                    "Call init_data_pipeline() first or pass state/contracts."
                )
            ingest_result = self.ingestor.ingest()
            state = ingest_result.state
            contracts = ingest_result.contracts
            self._questions = ingest_result.questions
            # Store headline-level data for per-contract predictions
            if ingest_result.headline_embeddings is not None:
                self._last_headline_embs = ingest_result.headline_embeddings
                self._last_headline_timestamps = ingest_result.headline_timestamps
                self._last_headline_texts = ingest_result.headline_texts

        # 2. Encode current state
        z = self.encode_state(
            state["news"], state["macro"], state["sentiment"]
        )

        # 3. OOD emergency check
        if self.ood_detector.is_ood(z):
            logger.error(
                "GLOBAL OOD — Mule Protection activated. "
                "Skipping all trades this cycle."
            )
            return []

        # 4. Mass scan: fast-predict all, score previous
        scan_result = self._scan_and_score(z, contracts)

        # 5. Deep eval: MCTS on top candidates
        max_candidates = self.config.scan_mcts_candidates
        candidates = scan_result.opportunities[:max_candidates]

        opportunities = []
        for pred in candidates:
            opp = self.evaluate_contract(z, pred.contract_id, pred.p_market)
            if opp is not None:
                opportunities.append(opp)

                # 6. Execute trade
                order = self.exchange.place_order(
                    contract_id=opp.contract_id,
                    side=(
                        OrderSide.BUY
                        if opp.direction == "BUY"
                        else OrderSide.SELL
                    ),
                    size=opp.position_fraction * 1000,
                    price=opp.p_market,
                )
                logger.info(
                    f"ORDER: {order.side.value} ${order.size:.2f} "
                    f"on {opp.question[:40]} @ {order.price:.3f} "
                    f"[{order.status.value}]"
                )

        if not opportunities:
            logger.info(
                f"Cycle {self._cycle_count}: "
                f"Scanned {scan_result.n_predictions}, "
                f"deep-eval'd {len(candidates)}, no tradeable alpha."
            )

        return opportunities

    def run_loop(self, interval_seconds: int = 900) -> None:
        """Run the unified loop on a timer.

        Each cycle: ingest → scan all → score previous → deep eval → trade.
        Ctrl+C to stop gracefully.
        """
        logger.info(
            f"Starting Radiant Seer loop — "
            f"every {interval_seconds}s, paper_mode={self.exchange.paper_mode}"
        )

        try:
            while True:
                cycle_start = time.monotonic()
                try:
                    opps = self.run_cycle()
                    if opps:
                        for opp in opps:
                            logger.info(
                                f"  ALPHA: {opp.direction} {opp.question[:40]} "
                                f"edge={opp.edge:+.3f} size={opp.position_fraction:.4f}"
                            )
                except Exception as e:
                    logger.error(f"Cycle {self._cycle_count} failed: {e}")

                elapsed = time.monotonic() - cycle_start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    logger.info(
                        f"Cycle {self._cycle_count} done in {elapsed:.1f}s. "
                        f"Next in {sleep_time:.0f}s."
                    )
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Radiant Seer stopped by user.")
            self._print_summary()

    def _print_summary(self) -> None:
        """Print session summary on shutdown."""
        pnl = self.exchange.get_pnl_summary()
        sc = self._scorecard

        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Cycles: {self._cycle_count}")

        # Trading summary
        logger.info(
            f"Orders: {pnl['total_orders']} "
            f"(buys={pnl.get('buys', 0)}, sells={pnl.get('sells', 0)})"
        )
        logger.info(f"Total size: ${pnl['total_size']:.2f}")

        # Scanner summary
        logger.info(
            f"Predictions: {sc.total_predictions} total, "
            f"{sc.total_scored} scored, "
            f"{sc.total_correct} correct ({sc.accuracy:.1%})"
        )
        logger.info(
            f"Simulated P&L (${self.config.scan_bet_size:.0f}/bet): "
            f"${sc.total_pnl_dollars:+,.2f}"
        )

        if sc.by_edge:
            logger.info("\nAccuracy by edge bucket:")
            for label, _, _ in EDGE_BUCKETS:
                if label in sc.by_edge:
                    b = sc.by_edge[label]
                    acc = (
                        b["correct"] / b["scored"]
                        if b["scored"] > 0
                        else 0.0
                    )
                    bucket_pnl = b.get("pnl_dollars", 0.0)
                    logger.info(
                        f"  {label:>6s}: {b['scored']:4d} scored, "
                        f"{b['correct']:4d} correct ({acc:.1%}), "
                        f"P&L ${bucket_pnl:+,.2f}"
                    )

        logger.info(f"\nScan log: {self._scan_log_path}")


def main() -> None:
    """CLI entrypoint for running Radiant Seer."""
    import argparse

    parser = argparse.ArgumentParser(description="Radiant Seer")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles (default: 300 = 5min)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to trained model weights",
    )
    args = parser.parse_args()

    seer = RadiantSeer()
    seer.load_models(
        Path(args.model_dir) if args.model_dir else None
    )
    seer.init_data_pipeline()

    if args.once:
        seer.run_cycle()
    else:
        seer.run_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
