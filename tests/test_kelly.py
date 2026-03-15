"""Tests for Kelly criterion position sizing."""

import pytest

from radiant_seer.execution.kelly_sizing import KellySizer


@pytest.fixture
def sizer():
    return KellySizer(kelly_fraction=0.25, max_risk=0.02, min_edge=0.05)


class TestKellySizer:
    def test_no_edge_no_position(self, sizer):
        result = sizer.compute(p_model=0.50, p_market=0.50)
        assert result.fraction == 0.0

    def test_small_edge_no_position(self, sizer):
        result = sizer.compute(p_model=0.53, p_market=0.50)
        assert result.fraction == 0.0  # Below min_edge of 0.05

    def test_buy_direction(self, sizer):
        result = sizer.compute(p_model=0.70, p_market=0.50)
        assert result.direction == "BUY"
        assert result.fraction > 0

    def test_sell_direction(self, sizer):
        result = sizer.compute(p_model=0.30, p_market=0.50)
        assert result.direction == "SELL"
        assert result.fraction > 0

    def test_max_risk_cap(self, sizer):
        # Large edge should be capped at max_risk
        result = sizer.compute(p_model=0.95, p_market=0.30)
        assert result.fraction <= 0.02
        assert result.capped

    def test_quarter_kelly_smaller_than_full(self):
        quarter = KellySizer(kelly_fraction=0.25, max_risk=1.0, min_edge=0.01)
        full = KellySizer(kelly_fraction=1.0, max_risk=1.0, min_edge=0.01)

        q_result = quarter.compute(p_model=0.70, p_market=0.50)
        f_result = full.compute(p_model=0.70, p_market=0.50)

        assert q_result.fraction < f_result.fraction

    def test_confidence_scaling(self):
        # Use a large max_risk so the cap doesn't mask the confidence effect
        sizer = KellySizer(kelly_fraction=0.25, max_risk=0.50, min_edge=0.05)
        high_conf = sizer.compute(p_model=0.60, p_market=0.50, confidence=1.0)
        low_conf = sizer.compute(p_model=0.60, p_market=0.50, confidence=0.5)

        assert high_conf.fraction > low_conf.fraction

    def test_extreme_market_prices(self, sizer):
        # Market at 0 — should not divide by zero
        result = sizer.compute(p_model=0.5, p_market=0.0)
        assert result.fraction >= 0

        # Market at 1 — should not divide by zero
        result = sizer.compute(p_model=0.5, p_market=1.0)
        assert result.fraction >= 0
