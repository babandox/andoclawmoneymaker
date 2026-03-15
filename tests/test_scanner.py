"""Tests for mass prediction scanner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.scanner import (
    EDGE_BUCKETS,
    MassScanner,
    Prediction,
    ScanResult,
    Scorecard,
    load_scan_log,
)


# ── Prediction dataclass ────────────────────────────────────────────


class TestPrediction:
    def test_creation(self):
        p = Prediction(
            timestamp="2024-01-01T00:00:00",
            contract_id="abc123",
            question="Will X happen?",
            p_model=0.65,
            p_market=0.50,
            edge=0.15,
            direction="BUY",
        )
        assert p.edge == 0.15
        assert p.direction == "BUY"
        assert not p.scored
        assert p.correct is None

    def test_sell_direction(self):
        p = Prediction(
            timestamp="2024-01-01T00:00:00",
            contract_id="abc123",
            question="Will X happen?",
            p_model=0.30,
            p_market=0.50,
            edge=-0.20,
            direction="SELL",
        )
        assert p.direction == "SELL"
        assert p.edge == -0.20


# ── Scorecard ────────────────────────────────────────────────────────


class TestScorecard:
    def test_empty(self):
        sc = Scorecard()
        assert sc.accuracy == 0.0
        assert sc.avg_profit_pct == 0.0
        assert sc.total_predictions == 0

    def test_record_prediction(self):
        sc = Scorecard()
        sc.record_prediction()
        sc.record_prediction()
        assert sc.total_predictions == 2

    def test_record_correct_score(self):
        sc = Scorecard()
        p = Prediction(
            timestamp="t",
            contract_id="c1",
            question="Q?",
            p_model=0.65,
            p_market=0.50,
            edge=0.15,
            direction="BUY",
            scored=True,
            correct=True,
            profit_pct=0.05,
            pnl_dollars=2.00,
        )
        sc.record_score(p)
        assert sc.total_scored == 1
        assert sc.total_correct == 1
        assert sc.accuracy == 1.0
        assert sc.avg_profit_pct == 0.05
        assert sc.total_pnl_dollars == 2.00

    def test_record_wrong_score(self):
        sc = Scorecard()
        p = Prediction(
            timestamp="t",
            contract_id="c1",
            question="Q?",
            p_model=0.65,
            p_market=0.50,
            edge=0.15,
            direction="BUY",
            scored=True,
            correct=False,
            profit_pct=-0.03,
        )
        sc.record_score(p)
        assert sc.total_scored == 1
        assert sc.total_correct == 0
        assert sc.accuracy == 0.0

    def test_edge_buckets(self):
        sc = Scorecard()
        # 7% edge → "5-10%" bucket
        p = Prediction(
            timestamp="t",
            contract_id="c1",
            question="Q?",
            p_model=0.57,
            p_market=0.50,
            edge=0.07,
            direction="BUY",
            scored=True,
            correct=True,
            profit_pct=0.03,
        )
        sc.record_score(p)
        assert "5-10%" in sc.by_edge
        assert sc.by_edge["5-10%"]["scored"] == 1
        assert sc.by_edge["5-10%"]["correct"] == 1

    def test_multiple_buckets(self):
        sc = Scorecard()
        for edge, correct in [(0.01, True), (0.03, False), (0.12, True)]:
            p = Prediction(
                timestamp="t",
                contract_id="c",
                question="Q?",
                p_model=0.5 + edge,
                p_market=0.5,
                edge=edge,
                direction="BUY",
                scored=True,
                correct=correct,
                profit_pct=edge if correct else -edge,
            )
            sc.record_score(p)
        assert sc.total_scored == 3
        assert sc.total_correct == 2
        assert "0-2%" in sc.by_edge
        assert "2-5%" in sc.by_edge
        assert "10-20%" in sc.by_edge

    def test_unscored_ignored(self):
        sc = Scorecard()
        p = Prediction(
            timestamp="t",
            contract_id="c",
            question="Q?",
            p_model=0.5,
            p_market=0.5,
            edge=0.0,
            direction="BUY",
            scored=False,
        )
        sc.record_score(p)
        assert sc.total_scored == 0


# ── MassScanner ──────────────────────────────────────────────────────


def _make_state():
    """Create a mock state dict."""
    return {
        "news": torch.randn(384),
        "macro": torch.randn(12),
        "sentiment": torch.tensor([0.5]),
    }


class TestMassScanner:
    def test_init(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )
        assert scanner._cycle_count == 0
        assert scanner.scorecard.total_predictions == 0

    def test_fast_predict_range(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )
        z = torch.randn(128)
        p = scanner._fast_predict(z)
        assert 0.0 <= p <= 1.0

    def test_encode_state(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )
        state = _make_state()
        z = scanner._encode_state(state)
        assert z.shape == (128,)

    def test_single_scan_cycle(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )

        state = _make_state()
        contracts = {"c1": 0.50, "c2": 0.30, "c3": 0.70}
        questions = {"c1": "Q1?", "c2": "Q2?", "c3": "Q3?"}

        result = scanner.scan_cycle(
            state=state, contracts=contracts, questions=questions
        )
        assert result.n_predictions == 3
        assert result.n_scored == 0  # first cycle, nothing to score
        assert scanner._cycle_count == 1
        assert scanner.scorecard.total_predictions == 3

    def test_scoring_across_cycles(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )

        state = _make_state()

        # Cycle 1: make predictions
        scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.50, "c2": 0.30},
            questions={"c1": "Q1?", "c2": "Q2?"},
        )
        assert len(scanner.pending) == 2

        # Cycle 2: prices changed → previous predictions get scored
        result2 = scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.55, "c2": 0.25},
            questions={"c1": "Q1?", "c2": "Q2?"},
        )
        assert result2.n_scored == 2
        assert scanner.scorecard.total_scored == 2

    def test_scoring_correctness(self, tmp_path):
        """Test that BUY predictions score correctly with dollar P&L."""
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )

        state = _make_state()

        # Force a known p_model by directly setting pending
        scanner.pending = {
            "c1": Prediction(
                timestamp="t0",
                contract_id="c1",
                question="Q?",
                p_model=0.70,
                p_market=0.50,
                edge=0.20,
                direction="BUY",
            ),
        }

        # Price went up (BUY was correct)
        result = scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.55},
            questions={"c1": "Q?"},
        )
        assert result.n_scored == 1
        assert result.n_correct == 1
        scored = result.scored_predictions[0]
        assert scored.correct is True
        assert scored.price_move == pytest.approx(0.05, abs=1e-6)
        # $20 bet, bought YES at 0.50, price moved +0.05
        # pnl = 20 * 0.05 / 0.50 = $2.00
        assert scored.pnl_dollars == pytest.approx(2.00, abs=0.01)
        assert result.cycle_pnl == pytest.approx(2.00, abs=0.01)

    def test_sell_scoring(self, tmp_path):
        """Test that SELL predictions score correctly with dollar P&L."""
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )

        state = _make_state()

        scanner.pending = {
            "c1": Prediction(
                timestamp="t0",
                contract_id="c1",
                question="Q?",
                p_model=0.30,
                p_market=0.50,
                edge=-0.20,
                direction="SELL",
            ),
        }

        # Price went down (SELL was correct)
        result = scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.45},
            questions={"c1": "Q?"},
        )
        assert result.n_correct == 1
        scored = result.scored_predictions[0]
        assert scored.correct is True
        assert scored.profit_pct == pytest.approx(0.05, abs=1e-6)
        # $20 bet, bought NO at (1-0.50)=0.50, price moved -0.05 (good for SELL)
        # pnl = 20 * 0.05 / 0.50 = $2.00
        assert scored.pnl_dollars == pytest.approx(2.00, abs=0.01)

    def test_skip_extreme_prices(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )

        state = _make_state()
        contracts = {
            "near_zero": 0.005,
            "near_one": 0.995,
            "normal": 0.50,
        }

        result = scanner.scan_cycle(
            state=state, contracts=contracts, questions={}
        )
        assert result.n_predictions == 1  # only 'normal'

    def test_opportunities_filtered_by_edge(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config,
            log_path=tmp_path / "test.jsonl",
            min_edge_to_surface=0.10,
        )

        state = _make_state()
        # Create contracts with varying distances from model's estimate
        # The model will produce the same p_model for all, so contracts
        # far from that value will have larger edges
        contracts = {f"c{i}": i / 20.0 for i in range(1, 20)}
        questions = {f"c{i}": f"Q{i}?" for i in range(1, 20)}

        result = scanner.scan_cycle(
            state=state, contracts=contracts, questions=questions
        )

        # Opportunities should only include those with |edge| >= 10%
        for opp in result.opportunities:
            assert abs(opp.edge) >= 0.10

    def test_opportunities_sorted_by_edge(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config,
            log_path=tmp_path / "test.jsonl",
            min_edge_to_surface=0.0,  # surface everything
        )

        state = _make_state()
        contracts = {"c1": 0.20, "c2": 0.50, "c3": 0.80}

        result = scanner.scan_cycle(
            state=state, contracts=contracts, questions={}
        )

        # Should be sorted by |edge| descending
        edges = [abs(o.edge) for o in result.opportunities]
        assert edges == sorted(edges, reverse=True)

    def test_disappeared_contract_not_scored(self, tmp_path):
        """Contracts that disappear between cycles aren't scored."""
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )

        state = _make_state()

        # Cycle 1
        scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.50, "c2": 0.30},
            questions={},
        )

        # Cycle 2: c2 disappeared
        result2 = scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.55},
            questions={},
        )
        assert result2.n_scored == 1  # only c1 scored

    def test_no_data_pipeline_raises(self, tmp_path):
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )
        with pytest.raises(RuntimeError, match="init_data_pipeline"):
            scanner.scan_cycle()


# ── Log file ─────────────────────────────────────────────────────────


class TestLogFile:
    def test_predictions_logged(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        config = SeerConfig(device="cpu")
        scanner = MassScanner(config=config, log_path=log_path)

        state = _make_state()
        scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.50},
            questions={"c1": "Will X?"},
        )

        assert log_path.exists()
        records = load_scan_log(log_path)
        assert len(records) == 1
        assert records[0]["type"] == "prediction"
        assert records[0]["contract_id"] == "c1"
        assert records[0]["question"] == "Will X?"

    def test_scores_logged(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        config = SeerConfig(device="cpu")
        scanner = MassScanner(config=config, log_path=log_path)

        state = _make_state()

        # Cycle 1
        scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.50},
            questions={"c1": "Q?"},
        )
        # Cycle 2
        scanner.scan_cycle(
            state=state,
            contracts={"c1": 0.55},
            questions={"c1": "Q?"},
        )

        records = load_scan_log(log_path)
        pred_records = [r for r in records if r["type"] == "prediction"]
        score_records = [r for r in records if r["type"] == "score"]
        assert len(pred_records) == 2  # one per cycle
        assert len(score_records) == 1  # scored after cycle 2
        assert score_records[0]["correct"] is not None

    def test_load_empty_log(self, tmp_path):
        records = load_scan_log(tmp_path / "nonexistent.jsonl")
        assert records == []

    def test_log_accumulates(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        config = SeerConfig(device="cpu")
        scanner = MassScanner(config=config, log_path=log_path)

        state = _make_state()
        contracts = {"c1": 0.50, "c2": 0.60}

        # 3 cycles
        for price_offset in [0.0, 0.02, -0.01]:
            adjusted = {k: v + price_offset for k, v in contracts.items()}
            scanner.scan_cycle(
                state=state, contracts=adjusted, questions={}
            )

        records = load_scan_log(log_path)
        # 3 cycles × 2 contracts = 6 predictions + 2 cycles of scoring × 2 = 4 scores
        pred_count = sum(1 for r in records if r["type"] == "prediction")
        score_count = sum(1 for r in records if r["type"] == "score")
        assert pred_count == 6
        assert score_count == 4


# ── Scorecard report ─────────────────────────────────────────────────


class TestReport:
    def test_print_report_no_crash(self, tmp_path, capsys):
        """Report prints without errors even with no data."""
        config = SeerConfig(device="cpu")
        scanner = MassScanner(
            config=config, log_path=tmp_path / "test.jsonl"
        )
        scanner.print_report()  # Should not crash
