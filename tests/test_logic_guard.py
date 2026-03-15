"""Tests for the logic guard."""

from datetime import date

import pytest

from radiant_seer.planning.logic_guard import (
    ElectionDateRule,
    FedRateChangeRule,
    LogicGuard,
    MaxEventFrequencyRule,
    ProbabilityBoundsRule,
    TrajectoryStep,
)


@pytest.fixture
def guard():
    return LogicGuard()


class TestLogicGuard:
    def test_valid_trajectory(self, guard):
        trajectory = [
            TrajectoryStep(event_type=0, timestamp=date(2026, 6, 1)),
            TrajectoryStep(event_type=1, timestamp=date(2026, 7, 1)),
            TrajectoryStep(event_type=2, timestamp=date(2026, 8, 1)),
        ]
        result = guard.validate_trajectory(trajectory)
        assert result.valid
        assert len(result.violations) == 0

    def test_temporal_violation(self, guard):
        trajectory = [
            TrajectoryStep(event_type=0, timestamp=date(2026, 8, 1)),
            TrajectoryStep(event_type=1, timestamp=date(2026, 6, 1)),  # Before previous
        ]
        result = guard.validate_trajectory(trajectory)
        assert not result.valid
        assert any("timestamp" in v.lower() or "precedes" in v.lower() for v in result.violations)


class TestElectionDateRule:
    def test_valid_election(self):
        rule = ElectionDateRule()
        trajectory = [
            TrajectoryStep(event_type=6, timestamp=date(2026, 12, 1)),
        ]
        result = rule.check(trajectory)
        assert result.valid

    def test_early_election_rejected(self):
        rule = ElectionDateRule()
        trajectory = [
            TrajectoryStep(event_type=6, timestamp=date(2026, 1, 1)),
        ]
        result = rule.check(trajectory)
        assert not result.valid


class TestFedRateChangeRule:
    def test_normal_rate_change(self):
        rule = FedRateChangeRule()
        trajectory = [
            TrajectoryStep(event_type=3, metadata={"rate_change_bps": 25}),
        ]
        result = rule.check(trajectory)
        assert result.valid

    def test_excessive_rate_change(self):
        rule = FedRateChangeRule()
        trajectory = [
            TrajectoryStep(event_type=3, metadata={"rate_change_bps": 100}),
        ]
        result = rule.check(trajectory)
        assert not result.valid


class TestProbabilityBoundsRule:
    def test_valid_probability(self):
        rule = ProbabilityBoundsRule()
        trajectory = [
            TrajectoryStep(event_type=0, metadata={"probability": 0.7}),
        ]
        result = rule.check(trajectory)
        assert result.valid

    def test_invalid_probability(self):
        rule = ProbabilityBoundsRule()
        trajectory = [
            TrajectoryStep(event_type=0, metadata={"probability": 1.5}),
        ]
        result = rule.check(trajectory)
        assert not result.valid


class TestMaxEventFrequencyRule:
    def test_within_limit(self):
        rule = MaxEventFrequencyRule()
        trajectory = [TrajectoryStep(event_type=0) for _ in range(5)]
        result = rule.check(trajectory)
        assert result.valid

    def test_exceeds_limit(self):
        rule = MaxEventFrequencyRule()
        trajectory = [TrajectoryStep(event_type=0) for _ in range(6)]
        result = rule.check(trajectory)
        assert not result.valid
