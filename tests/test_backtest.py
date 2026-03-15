"""Tests for the backtesting engine."""

import json

import numpy as np
import pytest

from radiant_seer.backtesting.backtest import Backtester, BacktestResult, Trade


def _make_snapshot(idx: int, prices: dict[str, float]) -> dict:
    """Create a minimal valid snapshot for testing."""
    rng = np.random.RandomState(idx)
    emb = rng.randn(384).astype(float)
    emb = (emb / np.linalg.norm(emb)).tolist()
    return {
        "timestamp": f"2026-03-{10+idx:02d}T12:00:00",
        "news_embedding": emb,
        "macro_values": rng.randn(12).tolist(),
        "sentiment": float(rng.uniform(-0.5, 0.5)),
        "n_headlines": 10,
        "headlines": [f"headline_{i}" for i in range(10)],
        "contracts": prices,
        "questions": {k: f"Will {k} happen?" for k in prices},
    }


@pytest.fixture
def snapshot_dir(tmp_path):
    """Create a dir with 5 snapshots showing a market moving."""
    prices_over_time = [
        {"ABC": 0.50, "DEF": 0.30},
        {"ABC": 0.55, "DEF": 0.28},
        {"ABC": 0.60, "DEF": 0.25},
        {"ABC": 0.65, "DEF": 0.22},
        {"ABC": 0.70, "DEF": 0.20},
    ]
    for i, prices in enumerate(prices_over_time):
        snap = _make_snapshot(i, prices)
        path = tmp_path / f"snapshot_2026031{i}_120000.json"
        with open(path, "w") as f:
            json.dump(snap, f)
    return tmp_path


class TestBacktestResult:
    def test_empty_result(self):
        r = BacktestResult()
        assert r.n_trades == 0
        assert r.total_return == 0.0

    def test_trade_dataclass(self):
        t = Trade(
            timestamp="2026-03-10",
            contract_id="ABC",
            question="Will ABC?",
            direction="BUY",
            p_model=0.7,
            p_market=0.5,
            edge=0.2,
            size_fraction=0.01,
        )
        assert t.pnl == 0.0
        assert not t.resolved


class TestBacktester:
    def test_run_with_snapshots(self, snapshot_dir):
        bt = Backtester(initial_bankroll=10000.0)
        result = bt.run(snapshot_dir=snapshot_dir)

        assert result.snapshots_processed == 5
        assert result.contracts_evaluated > 0
        assert len(result.equity_curve) > 0

    def test_run_insufficient_data(self, tmp_path):
        # Only 1 snapshot — not enough
        snap = _make_snapshot(0, {"ABC": 0.5})
        path = tmp_path / "snapshot_20260310_120000.json"
        with open(path, "w") as f:
            json.dump(snap, f)

        bt = Backtester(initial_bankroll=10000.0)
        result = bt.run(snapshot_dir=tmp_path)
        assert result.n_trades == 0

    def test_metrics_computed(self, snapshot_dir):
        bt = Backtester(initial_bankroll=10000.0)
        result = bt.run(snapshot_dir=snapshot_dir)

        # Metrics should be populated if trades happened
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert 0.0 <= result.max_drawdown <= 1.0
        assert isinstance(result.win_rate, float)

    def test_print_report_no_crash(self, snapshot_dir):
        bt = Backtester(initial_bankroll=10000.0)
        result = bt.run(snapshot_dir=snapshot_dir)
        # Should not raise
        bt.print_report(result)

    def test_calibration_bins(self, snapshot_dir):
        bt = Backtester(initial_bankroll=10000.0)
        result = bt.run(snapshot_dir=snapshot_dir)

        if result.calibration_bins:
            for b in result.calibration_bins:
                assert "mean_predicted" in b
                assert "mean_actual" in b
                assert b["count"] > 0
