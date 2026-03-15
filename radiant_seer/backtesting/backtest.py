"""Phase 7: Backtesting engine.

Replays historical snapshots through the full pipeline, tracks simulated PnL,
and computes performance metrics (Sharpe, max drawdown, calibration).

Usage:
    python -m radiant_seer.backtesting.backtest
    python -m radiant_seer.backtesting.backtest --model-dir data/models
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.data_swarm.collector import load_snapshots
from radiant_seer.execution.kelly_sizing import KellySizer
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.planning.reward_module import OutcomeDecoder


@dataclass
class Trade:
    timestamp: str
    contract_id: str
    question: str
    direction: str
    p_model: float
    p_market: float
    edge: float
    size_fraction: float
    resolved: bool = False
    outcome: float | None = None  # 1.0 if YES, 0.0 if NO
    pnl: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    snapshots_processed: int = 0
    contracts_evaluated: int = 0

    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0

    # Calibration
    calibration_bins: list[dict] = field(default_factory=list)


class Backtester:
    """Replay historical data through the prediction pipeline."""

    def __init__(
        self,
        config: SeerConfig | None = None,
        model_dir: Path | None = None,
        initial_bankroll: float = 10000.0,
    ):
        self.config = config or SeerConfig()
        self.device = self.config.device
        self.model_dir = model_dir or (
            self.config.project_root / "data" / "models"
        )
        self.initial_bankroll = initial_bankroll

        # Load models
        self.encoder = MultimodalEncoder(
            news_dim=self.config.news_embedding_dim,
            macro_dim=self.config.macro_feature_count,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.decoder = OutcomeDecoder(
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self._load_models()

        self.kelly = KellySizer(
            kelly_fraction=self.config.kelly_fraction,
            max_risk=self.config.max_portfolio_risk,
            min_edge=self.config.alpha_threshold,
        )

    def _load_models(self) -> None:
        # Prefer fine-tuned encoder if available
        ft_path = self.model_dir / "encoder_finetuned.pt"
        enc_path = self.model_dir / "encoder_vicreg.pt"
        dec_path = self.model_dir / "outcome_decoder.pt"

        if ft_path.exists():
            self.encoder.load_state_dict(
                torch.load(ft_path, weights_only=True)
            )
            logger.info("Loaded fine-tuned encoder")
        elif enc_path.exists():
            self.encoder.load_state_dict(
                torch.load(enc_path, weights_only=True)
            )
            logger.info("Loaded base encoder")

        if dec_path.exists():
            self.decoder.load_state_dict(
                torch.load(dec_path, weights_only=True)
            )
            logger.info("Loaded outcome decoder")

        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def _encode_snapshot(self, snap: dict) -> torch.Tensor:
        """Encode a single snapshot to latent z."""
        news = torch.tensor(
            [snap["news_embedding"]], dtype=torch.float32
        ).to(self.device)
        macro = torch.tensor(
            [snap["macro_values"]], dtype=torch.float32
        ).to(self.device)
        sentiment = torch.tensor(
            [[snap.get("sentiment", 0.0)]], dtype=torch.float32
        ).to(self.device)
        return self.encoder(news, macro, sentiment).squeeze(0)

    @torch.no_grad()
    def _get_p_model(self, z: torch.Tensor) -> float:
        """Get model probability from latent state."""
        return self.decoder(z.unsqueeze(0)).item()

    def run(
        self,
        snapshot_dir: Path | None = None,
        resolve_func: callable | None = None,
    ) -> BacktestResult:
        """Run backtest over historical snapshots.

        Args:
            snapshot_dir: Directory containing snapshot JSON files.
            resolve_func: Optional function(contract_id, snap_t, snap_t1) → float|None
                that resolves contract outcomes between snapshots.
                If None, uses price-change heuristic.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        snapshots = load_snapshots(snapshot_dir)
        if len(snapshots) < 2:
            logger.error("Need at least 2 snapshots for backtesting")
            return BacktestResult()

        logger.info(f"Backtesting on {len(snapshots)} snapshots")

        bankroll = self.initial_bankroll
        equity_curve = [bankroll]
        all_trades: list[Trade] = []
        contracts_evaluated = 0
        predictions: list[tuple[float, float]] = []  # (p_model, actual)

        for i in range(len(snapshots) - 1):
            snap_t = snapshots[i]
            snap_t1 = snapshots[i + 1]
            ts = snap_t.get("timestamp", f"snap_{i}")

            if snap_t.get("n_headlines", 0) == 0:
                equity_curve.append(bankroll)
                continue

            z = self._encode_snapshot(snap_t)
            contracts = snap_t.get("contracts", {})
            questions = snap_t.get("questions", {})
            next_contracts = snap_t1.get("contracts", {})

            for cid, p_market in contracts.items():
                if p_market <= 0.01 or p_market >= 0.99:
                    continue  # Skip near-certain contracts
                contracts_evaluated += 1

                p_model = self._get_p_model(z)
                edge = p_model - p_market

                position = self.kelly.compute(
                    p_model=p_model,
                    p_market=p_market,
                )

                if position.fraction <= 0:
                    continue

                # Resolve: use next snapshot's price as proxy for outcome
                if resolve_func:
                    outcome = resolve_func(cid, snap_t, snap_t1)
                else:
                    outcome = self._price_change_resolve(
                        cid, p_market, next_contracts
                    )

                if outcome is None:
                    continue

                # Compute PnL
                size = position.fraction * bankroll
                if position.direction == "BUY":
                    # Bought YES at p_market, resolved to outcome
                    pnl = size * (outcome - p_market) / p_market
                else:
                    # Sold YES (bought NO) at (1-p_market)
                    pnl = size * (p_market - outcome) / (1 - p_market)

                trade = Trade(
                    timestamp=ts,
                    contract_id=cid,
                    question=questions.get(cid, cid[:20]),
                    direction=position.direction,
                    p_model=p_model,
                    p_market=p_market,
                    edge=edge,
                    size_fraction=position.fraction,
                    resolved=True,
                    outcome=outcome,
                    pnl=pnl,
                )
                all_trades.append(trade)
                bankroll += pnl
                predictions.append((p_model, outcome))

            equity_curve.append(bankroll)

        # Compute metrics
        result = BacktestResult(
            trades=all_trades,
            equity_curve=equity_curve,
            snapshots_processed=len(snapshots),
            contracts_evaluated=contracts_evaluated,
            n_trades=len(all_trades),
        )

        if all_trades:
            self._compute_metrics(result, predictions)

        return result

    def _price_change_resolve(
        self,
        contract_id: str,
        p_market_t: float,
        next_contracts: dict[str, float],
    ) -> float | None:
        """Use next snapshot's price as a proxy resolution.

        If price moved toward 1.0 → treat as partial YES.
        If price moved toward 0.0 → treat as partial NO.
        """
        if contract_id not in next_contracts:
            return None
        return next_contracts[contract_id]

    def _compute_metrics(
        self,
        result: BacktestResult,
        predictions: list[tuple[float, float]],
    ) -> None:
        """Compute performance and calibration metrics."""
        trades = result.trades
        equity = result.equity_curve

        # Total return
        result.total_return = (equity[-1] / equity[0]) - 1.0

        # Win rate
        wins = sum(1 for t in trades if t.pnl > 0)
        result.win_rate = wins / len(trades) if trades else 0.0

        # Sharpe ratio (from equity curve returns)
        if len(equity) > 2:
            returns = np.diff(equity) / np.array(equity[:-1])
            if returns.std() > 0:
                result.sharpe_ratio = (
                    returns.mean() / returns.std() * np.sqrt(252)
                )

        # Max drawdown
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

        # Calibration (reliability diagram)
        if predictions:
            result.calibration_bins = self._calibration(predictions)

    def _calibration(
        self, predictions: list[tuple[float, float]], n_bins: int = 10
    ) -> list[dict]:
        """Compute calibration bins for reliability diagram."""
        bins: list[dict] = []
        edges = np.linspace(0, 1, n_bins + 1)

        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = [
                (p, a) for p, a in predictions if lo <= p < hi
            ]
            if in_bin:
                mean_pred = np.mean([p for p, _ in in_bin])
                mean_actual = np.mean([a for _, a in in_bin])
                bins.append({
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "mean_predicted": float(mean_pred),
                    "mean_actual": float(mean_actual),
                    "count": len(in_bin),
                })

        return bins

    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted backtest report."""
        logger.info("=" * 60)
        logger.info("BACKTEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Snapshots: {result.snapshots_processed}")
        logger.info(f"Contracts evaluated: {result.contracts_evaluated}")
        logger.info(f"Trades: {result.n_trades}")
        logger.info(f"Total return: {result.total_return:+.2%}")
        logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Win rate: {result.win_rate:.2%}")
        logger.info(
            f"Final equity: ${result.equity_curve[-1]:,.2f} "
            f"(from ${result.equity_curve[0]:,.2f})"
        )

        if result.calibration_bins:
            logger.info("\nCalibration:")
            for b in result.calibration_bins:
                logger.info(
                    f"  [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}] "
                    f"predicted={b['mean_predicted']:.3f} "
                    f"actual={b['mean_actual']:.3f} "
                    f"n={b['count']}"
                )

        if result.trades:
            logger.info("\nTop 5 trades by |PnL|:")
            sorted_trades = sorted(
                result.trades, key=lambda t: abs(t.pnl), reverse=True
            )
            for t in sorted_trades[:5]:
                logger.info(
                    f"  {t.direction} {t.question[:35]:35s} "
                    f"edge={t.edge:+.3f} pnl=${t.pnl:+.2f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Radiant Seer")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--bankroll", type=float, default=10000.0)
    args = parser.parse_args()

    bt = Backtester(
        model_dir=Path(args.model_dir) if args.model_dir else None,
        initial_bankroll=args.bankroll,
    )
    result = bt.run(
        snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
    )
    bt.print_report(result)


if __name__ == "__main__":
    main()
