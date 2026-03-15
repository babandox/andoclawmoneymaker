"""Mass prediction scanner — predict everything, score everything, learn.

Continuously scans all active Polymarket contracts:
  1. Fast-predict: encoder + decoder → p_model (no MCTS, instant)
  2. Score: compare previous cycle's predictions to current market prices
  3. Log: append every prediction + score to JSONL for offline analysis
  4. Surface: flag contracts where model sees significant edge

The feedback loop is the point — the model learns what it's good at
by seeing which predicted mispricings actually corrected.

Usage:
    python -m radiant_seer.scanner               # single scan
    python -m radiant_seer.scanner --loop 300     # every 5 min
    python -m radiant_seer.scanner --report       # analyze past predictions
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
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
from radiant_seer.intelligence.contract_decoder import ContractDecoder
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.intelligence.relevance import (
    CausalDomainGraph,
    LearnedRelevanceScorer,
    RelevanceRouter,
)
from radiant_seer.planning.reward_module import OutcomeDecoder

# Edge buckets for accuracy breakdown
EDGE_BUCKETS = [
    ("0-2%", 0.0, 0.02),
    ("2-5%", 0.02, 0.05),
    ("5-10%", 0.05, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20%+", 0.20, 1.0),
]


@dataclass
class Prediction:
    """A single prediction record."""

    timestamp: str
    contract_id: str
    question: str
    p_model: float
    p_market: float
    edge: float  # p_model - p_market
    direction: str  # BUY or SELL
    bet_size: float = 20.0  # per-prediction bet size (Kelly or flat)

    # Scoring (filled after next cycle)
    scored: bool = False
    score_timestamp: str | None = None
    p_market_after: float | None = None
    price_move: float | None = None  # p_market_after - p_market
    correct: bool | None = None  # price moved in predicted direction
    profit_pct: float | None = None  # signed profit per unit
    pnl_dollars: float | None = None  # dollar P&L on simulated bet


@dataclass
class ScanResult:
    """Result of one scan cycle."""

    timestamp: str
    n_contracts: int
    n_predictions: int
    n_scored: int
    n_correct: int
    accuracy: float  # n_correct / n_scored (or 0)
    cycle_pnl: float = 0.0  # dollar P&L this cycle
    opportunities: list[Prediction] = field(default_factory=list)
    scored_predictions: list[Prediction] = field(default_factory=list)


@dataclass
class Scorecard:
    """Running accuracy tracker across scan cycles."""

    total_predictions: int = 0
    total_scored: int = 0
    total_correct: int = 0
    total_profit_pct: float = 0.0
    total_pnl_dollars: float = 0.0  # cumulative dollar P&L

    # By edge bucket: label → {scored, correct, profit, pnl_dollars}
    by_edge: dict[str, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.total_correct / self.total_scored

    @property
    def avg_profit_pct(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.total_profit_pct / self.total_scored

    def to_dict(self) -> dict:
        """Serialize scorecard to dict for saving."""
        return {
            "total_predictions": self.total_predictions,
            "total_scored": self.total_scored,
            "total_correct": self.total_correct,
            "total_profit_pct": self.total_profit_pct,
            "total_pnl_dollars": self.total_pnl_dollars,
            "by_edge": self.by_edge,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scorecard":
        """Restore scorecard from saved dict."""
        sc = cls()
        sc.total_predictions = data.get("total_predictions", 0)
        sc.total_scored = data.get("total_scored", 0)
        sc.total_correct = data.get("total_correct", 0)
        sc.total_profit_pct = data.get("total_profit_pct", 0.0)
        sc.total_pnl_dollars = data.get("total_pnl_dollars", 0.0)
        sc.by_edge = data.get("by_edge", {})
        return sc

    @classmethod
    def from_scan_log(cls, log_path: Path) -> "Scorecard":
        """Rebuild scorecard from the JSONL scan log on disk.

        Replays all 'score' and 'prediction' records so P&L, accuracy,
        and edge-bucket breakdowns survive a crash/restart.
        """
        sc = cls()
        if not log_path.exists():
            return sc

        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "prediction":
                    sc.total_predictions += 1
                elif rec.get("type") == "score":
                    sc.total_scored += 1
                    pnl = rec.get("pnl_dollars", 0.0)
                    profit = rec.get("profit_pct", 0.0)
                    correct = rec.get("correct", False)
                    if correct:
                        sc.total_correct += 1
                    sc.total_profit_pct += profit
                    sc.total_pnl_dollars += pnl

                    # Edge bucket
                    abs_edge = abs(rec.get("edge", 0.0))
                    for label, lo, hi in EDGE_BUCKETS:
                        if lo <= abs_edge < hi:
                            if label not in sc.by_edge:
                                sc.by_edge[label] = {
                                    "scored": 0,
                                    "correct": 0,
                                    "profit": 0.0,
                                    "pnl_dollars": 0.0,
                                }
                            bucket = sc.by_edge[label]
                            bucket["scored"] += 1
                            if correct:
                                bucket["correct"] += 1
                            bucket["profit"] += profit
                            bucket["pnl_dollars"] += pnl
                            break

        logger.info(
            f"Restored scorecard from log: {sc.total_scored:,} scored, "
            f"{sc.total_correct:,} correct ({sc.accuracy:.1%}), "
            f"P&L ${sc.total_pnl_dollars:+,.2f}"
        )
        return sc

    def record_prediction(self) -> None:
        self.total_predictions += 1

    def record_score(self, prediction: Prediction) -> None:
        if not prediction.scored:
            return
        self.total_scored += 1
        if prediction.correct:
            self.total_correct += 1
        self.total_profit_pct += prediction.profit_pct or 0.0
        self.total_pnl_dollars += prediction.pnl_dollars or 0.0

        # Bucket by absolute edge
        abs_edge = abs(prediction.edge)
        for label, lo, hi in EDGE_BUCKETS:
            if lo <= abs_edge < hi:
                if label not in self.by_edge:
                    self.by_edge[label] = {
                        "scored": 0,
                        "correct": 0,
                        "profit": 0.0,
                        "pnl_dollars": 0.0,
                    }
                bucket = self.by_edge[label]
                bucket["scored"] += 1
                if prediction.correct:
                    bucket["correct"] += 1
                bucket["profit"] += prediction.profit_pct or 0.0
                bucket["pnl_dollars"] += prediction.pnl_dollars or 0.0
                break


class MassScanner:
    """Scan all markets, predict fair value, score against price movements."""

    def __init__(
        self,
        config: SeerConfig | None = None,
        model_dir: Path | None = None,
        log_path: Path | None = None,
        min_edge_to_surface: float = 0.05,
    ):
        self.config = config or SeerConfig()
        self.device = self.config.device
        self.min_edge = min_edge_to_surface
        self.log_path = log_path or (self.config.data_dir / "scan_log.jsonl")

        # Lightweight models — encoder + decoder only (no MCTS)
        self.encoder = MultimodalEncoder(
            news_dim=self.config.news_embedding_dim,
            macro_dim=self.config.macro_feature_count,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.decoder = OutcomeDecoder(
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        # Per-contract models
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

        self._load_models(model_dir)

        # Data pipeline
        self.normalizer = StateNormalizer(
            macro_dim=self.config.macro_feature_count,
            news_dim=self.config.news_embedding_dim,
        )
        self.ingestor: DataIngestor | None = None

        # News embedder for contract question embedding
        self._news_embedder: NewsEmbedder | None = None
        self._contract_emb_cache: dict[str, np.ndarray] = {}

        # State — restore from scan log if it exists
        self.scorecard = Scorecard.from_scan_log(self.log_path) if self.log_path.exists() else Scorecard()
        self.pending = self._restore_pending() if self.log_path.exists() else {}
        self._cycle_count = 0

    def _load_models(self, model_dir: Path | None = None) -> None:
        model_dir = model_dir or (self.config.project_root / "data" / "models")

        # Prefer fine-tuned encoder
        for name in ["encoder_finetuned.pt", "encoder_vicreg.pt"]:
            path = model_dir / name
            if path.exists():
                self.encoder.load_state_dict(
                    torch.load(path, weights_only=True)
                )
                logger.info(f"Loaded {name}")
                break

        dec_path = model_dir / "outcome_decoder.pt"
        if dec_path.exists():
            self.decoder.load_state_dict(
                torch.load(dec_path, weights_only=True)
            )
            logger.info("Loaded outcome decoder")

        # Per-contract model weights (optional)
        cd_path = model_dir / "contract_decoder.pt"
        if cd_path.exists():
            self.contract_decoder.load_state_dict(
                torch.load(cd_path, weights_only=True)
            )
            logger.info("Loaded contract decoder")

        rs_path = model_dir / "relevance_scorer.pt"
        if rs_path.exists():
            self.relevance_scorer.load_state_dict(
                torch.load(rs_path, weights_only=True)
            )
            logger.info("Loaded relevance scorer")

        self.encoder.eval()
        self.decoder.eval()
        self.contract_decoder.eval()
        self.relevance_scorer.eval()

    def _get_contract_emb(self, question: str) -> np.ndarray:
        """Get or compute and cache a contract question embedding."""
        if question in self._contract_emb_cache:
            return self._contract_emb_cache[question]
        if self._news_embedder is None:
            self._news_embedder = NewsEmbedder(dim=self.config.news_embedding_dim)
        emb = self._news_embedder.embed([question])[0]
        self._contract_emb_cache[question] = emb
        return emb

    def init_data_pipeline(
        self, fred_api_key: str | None = None
    ) -> None:
        """Initialize data pipeline with unfiltered Polymarket scraper."""
        from radiant_seer.data_swarm.scrapers.polymarket_scraper import (
            PolymarketScraper,
        )

        self.ingestor = DataIngestor(
            normalizer=self.normalizer,
            fred_api_key=fred_api_key or os.environ.get("FRED_API_KEY"),
            contract_ids=[],
            news_dim=self.config.news_embedding_dim,
        )
        # Disable strict regex filter — keep all contracts from relevant tags
        self.ingestor.polymarket = PolymarketScraper(
            apply_filter=False,
        )
        logger.info("Scanner data pipeline initialized (unfiltered within tags)")

    @torch.no_grad()
    def _fast_predict(self, z: Tensor) -> float:
        """Fast prediction: latent state → decoder → probability."""
        return self.decoder(z.unsqueeze(0)).item()

    @torch.no_grad()
    def _encode_state(self, state: dict[str, Tensor]) -> Tensor:
        """Encode raw state inputs to latent vector."""
        return self.encoder(
            state["news"].unsqueeze(0).to(self.device),
            state["macro"].unsqueeze(0).to(self.device),
            state["sentiment"].unsqueeze(0).to(self.device),
        ).squeeze(0)

    def scan_cycle(
        self,
        state: dict[str, Tensor] | None = None,
        contracts: dict[str, float] | None = None,
        questions: dict[str, str] | None = None,
    ) -> ScanResult:
        """Run one scan cycle: predict all, score previous, log.

        Args:
            state: Pre-built state tensors (news, macro, sentiment).
                If None, ingests live data automatically.
            contracts: Dict of contract_id → market price.
            questions: Dict of contract_id → question text.

        Returns:
            ScanResult with accuracy stats and opportunities.
        """
        self._cycle_count += 1
        ts = datetime.now().isoformat()

        # Ingest live data if not provided
        headline_embs = None
        headline_timestamps: list[float] = []
        headline_texts: list[str] = []
        liquidity: dict[str, float] = {}
        if state is None or contracts is None:
            if self.ingestor is None:
                raise RuntimeError(
                    "Call init_data_pipeline() first or pass state/contracts."
                )
            ingest_result = self.ingestor.ingest()
            state = ingest_result.state
            contracts = ingest_result.contracts
            questions = ingest_result.questions
            headline_embs = ingest_result.headline_embeddings
            headline_timestamps = ingest_result.headline_timestamps
            headline_texts = ingest_result.headline_texts
            liquidity = ingest_result.liquidity
        questions = questions or {}

        # Filter out low-liquidity contracts (dead markets that never move)
        min_liq = self.config.scan_min_liquidity
        if liquidity and min_liq > 0:
            n_before = len(contracts)
            contracts = {
                cid: price
                for cid, price in contracts.items()
                if liquidity.get(cid, 0) >= min_liq
            }
            n_filtered = n_before - len(contracts)
            if n_filtered > 0:
                logger.info(
                    f"Liquidity filter: kept {len(contracts)}/{n_before} "
                    f"contracts (dropped {n_filtered} with <${min_liq:,.0f} liquidity)"
                )

        # 1. Score previous cycle's predictions against current prices
        scored = self._score_previous(contracts)

        # 2. Encode current state
        z = self._encode_state(state)

        # Determine if we can do per-contract predictions
        use_per_contract = (
            headline_embs is not None
            and len(headline_texts) > 0
        )

        if use_per_contract:
            h_embs_t = torch.tensor(
                headline_embs, dtype=torch.float32
            ).to(self.device)
            macro_t = state["macro"].unsqueeze(0).to(self.device)
            sent_t = state["sentiment"].unsqueeze(0).to(self.device)
            if sent_t.dim() == 1:
                sent_t = sent_t.unsqueeze(0)

            with torch.no_grad():
                context_z, headline_tokens = self.encoder.forward_with_headlines(
                    h_embs_t.unsqueeze(0), macro_t, sent_t,
                )
        else:
            p_model = self._fast_predict(z)

        # 3. Predict all contracts
        new_predictions: dict[str, Prediction] = {}
        opportunities: list[Prediction] = []

        # Pre-compute headline tags ONCE
        cached_headline_tags = None
        if use_per_contract:
            cached_headline_tags = self.relevance_router.cache_headline_tags(
                headline_texts
            )

        for cid, p_market in contracts.items():
            # Skip near-certain contracts (no meaningful edge)
            if p_market <= 0.01 or p_market >= 0.99:
                continue

            if use_per_contract:
                question = questions.get(cid, "")
                c_emb_np = self._get_contract_emb(question)
                c_emb_t = torch.tensor(
                    c_emb_np, dtype=torch.float32
                ).to(self.device)

                weights = self.relevance_router.compute_weights(
                    headline_embs=h_embs_t,
                    headline_timestamps=headline_timestamps,
                    contract_emb=c_emb_t,
                    headline_texts=headline_texts,
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
                    question=questions.get(cid, ""),
                    p_model=round(p_model, 6),
                    p_market=round(p_market, 6),
                    edge=round(edge, 6),
                    direction="BUY" if edge > 0 else "SELL",
                )

            new_predictions[cid] = pred
            self.scorecard.record_prediction()
            self._log_prediction(pred)

            if abs(edge) >= self.min_edge:
                opportunities.append(pred)

        # 4. Store as pending for next cycle's scoring
        self.pending = new_predictions

        n_correct = sum(1 for s in scored if s.correct)
        accuracy = n_correct / len(scored) if scored else 0.0
        cycle_pnl = sum(s.pnl_dollars or 0.0 for s in scored)

        if scored:
            logger.info(
                f"Scan #{self._cycle_count}: {len(new_predictions)} predictions, "
                f"{len(scored)} scored ({n_correct} correct = {accuracy:.1%}), "
                f"cycle P&L ${cycle_pnl:+,.2f} | "
                f"cumulative ${self.scorecard.total_pnl_dollars:+,.2f}"
            )
        else:
            logger.info(
                f"Scan #{self._cycle_count}: {len(new_predictions)} predictions "
                f"(first cycle, nothing to score yet)"
            )

        return ScanResult(
            timestamp=ts,
            n_contracts=len(contracts),
            n_predictions=len(new_predictions),
            n_scored=len(scored),
            n_correct=n_correct,
            accuracy=accuracy,
            cycle_pnl=cycle_pnl,
            opportunities=sorted(
                opportunities, key=lambda p: abs(p.edge), reverse=True
            ),
            scored_predictions=scored,
        )

    def _score_previous(
        self, current_prices: dict[str, float]
    ) -> list[Prediction]:
        """Score previous cycle's predictions against current prices."""
        bet = self.config.scan_bet_size
        scored = []
        for cid, pred in self.pending.items():
            if cid not in current_prices:
                continue

            p_now = current_prices[cid]
            price_move = p_now - pred.p_market
            direction_sign = 1.0 if pred.direction == "BUY" else -1.0
            profit = price_move * direction_sign

            # Dollar P&L on simulated $bet
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

            self.scorecard.record_score(pred)
            self._log_score(pred)
            scored.append(pred)

        return scored

    def _restore_pending(self) -> dict[str, Prediction]:
        """Restore the last batch of unscored predictions from the scan log.

        Reads the log backwards to find the most recent prediction timestamp,
        then loads all predictions from that timestamp as pending so the
        first cycle after restart can score them.
        """
        if not self.log_path.exists():
            return {}

        # Find the last prediction timestamp and collect those predictions
        last_ts = None
        pending: dict[str, Prediction] = {}

        # Read all prediction records, keep only the last batch
        with open(self.log_path) as f:
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

    def _log_prediction(self, pred: Prediction) -> None:
        """Append prediction to JSONL log."""
        self._append_log({
            "type": "prediction",
            "timestamp": pred.timestamp,
            "contract_id": pred.contract_id,
            "question": pred.question,
            "p_model": pred.p_model,
            "p_market": pred.p_market,
            "edge": pred.edge,
            "direction": pred.direction,
        })

    def _log_score(self, pred: Prediction) -> None:
        """Append score to JSONL log."""
        self._append_log({
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
            "bet_size": self.config.scan_bet_size,
        })

    def _append_log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_report(self) -> None:
        """Print running accuracy report."""
        sc = self.scorecard
        logger.info("=" * 60)
        logger.info("MASS SCANNER REPORT")
        logger.info("=" * 60)
        logger.info(f"Cycles: {self._cycle_count}")
        logger.info(f"Total predictions: {sc.total_predictions}")
        logger.info(f"Scored: {sc.total_scored}")
        logger.info(f"Correct: {sc.total_correct} ({sc.accuracy:.1%})")
        logger.info(
            f"Simulated P&L (${self.config.scan_bet_size:.0f}/bet): "
            f"${sc.total_pnl_dollars:+,.2f}"
        )

        if sc.by_edge:
            logger.info("\nAccuracy by edge bucket:")
            for label, _, _ in EDGE_BUCKETS:
                if label in sc.by_edge:
                    b = sc.by_edge[label]
                    acc = b["correct"] / b["scored"] if b["scored"] > 0 else 0.0
                    bucket_pnl = b.get("pnl_dollars", 0.0)
                    logger.info(
                        f"  {label:>6s}: {b['scored']:4d} scored, "
                        f"{b['correct']:4d} correct ({acc:.1%}), "
                        f"P&L ${bucket_pnl:+,.2f}"
                    )

    def run_loop(self, interval_seconds: int = 300) -> None:
        """Run continuous scanning loop.

        Args:
            interval_seconds: Seconds between scan cycles (default 5 min).
        """
        logger.info(
            f"Starting mass scanner — every {interval_seconds}s"
        )
        try:
            while True:
                start = time.monotonic()
                try:
                    self.scan_cycle()
                except Exception as e:
                    logger.error(f"Scan cycle {self._cycle_count} failed: {e}")

                elapsed = time.monotonic() - start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    logger.info(
                        f"Cycle {self._cycle_count} done in {elapsed:.1f}s. "
                        f"Next in {sleep_time:.0f}s."
                    )
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Scanner stopped.")
            self.print_report()


def load_scan_log(log_path: Path | None = None) -> list[dict]:
    """Load scan log from JSONL file.

    Returns:
        List of log records (predictions and scores).
    """
    log_path = log_path or (
        Path(__file__).resolve().parent.parent / "data" / "scan_log.jsonl"
    )
    if not log_path.exists():
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_log(log_path: Path | None = None) -> None:
    """Analyze scan log and print summary statistics."""
    records = load_scan_log(log_path)
    if not records:
        logger.info("No scan log found.")
        return

    predictions = [r for r in records if r["type"] == "prediction"]
    scores = [r for r in records if r["type"] == "score"]

    logger.info("=" * 60)
    logger.info("SCAN LOG ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total prediction records: {len(predictions)}")
    logger.info(f"Total score records: {len(scores)}")

    if not scores:
        logger.info("No scored predictions yet.")
        return

    correct = sum(1 for s in scores if s.get("correct"))
    accuracy = correct / len(scores)
    avg_profit = sum(s.get("profit_pct", 0) for s in scores) / len(scores)

    logger.info(f"Overall accuracy: {correct}/{len(scores)} = {accuracy:.1%}")
    logger.info(f"Average profit per prediction: {avg_profit:+.6f}")

    # Breakdown by edge bucket
    logger.info("\nBy edge bucket:")
    for label, lo, hi in EDGE_BUCKETS:
        bucket = [s for s in scores if lo <= abs(s.get("edge", 0)) < hi]
        if bucket:
            bc = sum(1 for s in bucket if s.get("correct"))
            ba = bc / len(bucket)
            bp = sum(s.get("profit_pct", 0) for s in bucket) / len(bucket)
            logger.info(
                f"  {label:>6s}: {len(bucket):4d} scored, "
                f"{bc:4d} correct ({ba:.1%}), "
                f"avg profit {bp:+.6f}"
            )

    # Top contracts by frequency
    from collections import Counter

    cid_counts = Counter(s["contract_id"] for s in scores)
    logger.info(f"\nUnique contracts scored: {len(cid_counts)}")
    logger.info("Top 10 most-predicted contracts:")
    for cid, count in cid_counts.most_common(10):
        contract_scores = [s for s in scores if s["contract_id"] == cid]
        cc = sum(1 for s in contract_scores if s.get("correct"))
        q = contract_scores[0].get("question", cid[:30])
        logger.info(f"  {q[:50]:50s} n={count} acc={cc/count:.0%}")


def main() -> None:
    """CLI entrypoint for the mass scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Mass prediction scanner")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between scans (default: 300 = 5min)",
    )
    parser.add_argument(
        "--once", action="store_true", help="Single scan and exit"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Analyze past scan log and exit",
    )
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Min edge to surface as opportunity (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    if args.report:
        analyze_log()
        return

    scanner = MassScanner(
        model_dir=Path(args.model_dir) if args.model_dir else None,
        min_edge_to_surface=args.min_edge,
    )
    scanner.init_data_pipeline()

    if args.once:
        scanner.scan_cycle()
        scanner.print_report()
    else:
        scanner.run_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
