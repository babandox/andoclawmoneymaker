"""Logic Guard: hard-coded rules to prune impossible MCTS branches.

Not a formal logic engine — 5-10 Python rules for initial markets.
Validates trajectories against known constraints (temporal, regulatory, physical).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class TrajectoryStep:
    event_type: int
    timestamp: date | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    valid: bool
    violations: list[str] = field(default_factory=list)


class LogicGuard:
    """Validates MCTS trajectories against hard constraints."""

    def __init__(self):
        self.rules: list[Rule] = [
            ElectionDateRule(),
            FedRateChangeRule(),
            TemporalCausalityRule(),
            ProbabilityBoundsRule(),
            MaxEventFrequencyRule(),
        ]

    def validate_trajectory(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        """Check a trajectory against all rules.

        Args:
            trajectory: Sequence of steps in an MCTS rollout.

        Returns:
            ValidationResult indicating if the trajectory is valid.
        """
        violations = []
        for rule in self.rules:
            result = rule.check(trajectory)
            if not result.valid:
                violations.extend(result.violations)

        return ValidationResult(valid=len(violations) == 0, violations=violations)

    def is_valid_transition(
        self, current_step: TrajectoryStep, next_step: TrajectoryStep
    ) -> ValidationResult:
        """Check if a single transition is valid (for pruning during MCTS expansion)."""
        return self.validate_trajectory([current_step, next_step])


class Rule:
    """Base class for validation rules."""

    def check(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        raise NotImplementedError


class ElectionDateRule(Rule):
    """Election cannot resolve before its mandated date."""

    # US election event types (by convention: event_type 6 = election resolution)
    ELECTION_EVENT = 6
    # Earliest possible resolution date for US presidential election
    EARLIEST_RESOLUTION = date(2026, 11, 3)  # Placeholder — update per actual market

    def check(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        for step in trajectory:
            if step.event_type == self.ELECTION_EVENT and step.timestamp is not None:
                if step.timestamp < self.EARLIEST_RESOLUTION:
                    return ValidationResult(
                        valid=False,
                        violations=[
                            f"Election resolution at {step.timestamp} is before "
                            f"mandated date {self.EARLIEST_RESOLUTION}"
                        ],
                    )
        return ValidationResult(valid=True)


class FedRateChangeRule(Rule):
    """Fed rate changes are bounded: max 50bps per meeting, 8 meetings/year."""

    MAX_CHANGE_BPS = 50
    # Event type 3 = rate change (by convention)
    RATE_EVENT = 3

    def check(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        violations = []
        for step in trajectory:
            if step.event_type == self.RATE_EVENT:
                change = step.metadata.get("rate_change_bps", 0)
                if abs(change) > self.MAX_CHANGE_BPS:
                    violations.append(
                        f"Rate change of {change}bps exceeds max {self.MAX_CHANGE_BPS}bps"
                    )
        return ValidationResult(valid=len(violations) == 0, violations=violations)


class TemporalCausalityRule(Rule):
    """Events must respect temporal ordering — no effect before cause."""

    def check(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        violations = []
        for i in range(1, len(trajectory)):
            prev, curr = trajectory[i - 1], trajectory[i]
            if prev.timestamp and curr.timestamp and curr.timestamp < prev.timestamp:
                violations.append(
                    f"Step {i}: timestamp {curr.timestamp} precedes "
                    f"previous step {prev.timestamp}"
                )
        return ValidationResult(valid=len(violations) == 0, violations=violations)


class ProbabilityBoundsRule(Rule):
    """Any probability metadata must be in [0, 1]."""

    def check(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        violations = []
        for i, step in enumerate(trajectory):
            for key in ("probability", "p_model", "p_market"):
                val = step.metadata.get(key)
                if val is not None and (val < 0 or val > 1):
                    violations.append(f"Step {i}: {key}={val} outside [0, 1]")
        return ValidationResult(valid=len(violations) == 0, violations=violations)


class MaxEventFrequencyRule(Rule):
    """No more than N events of same type in a short trajectory (prevents loops)."""

    MAX_SAME_TYPE = 5

    def check(self, trajectory: list[TrajectoryStep]) -> ValidationResult:
        counts: dict[int, int] = {}
        for step in trajectory:
            counts[step.event_type] = counts.get(step.event_type, 0) + 1

        violations = []
        for event_type, count in counts.items():
            if count > self.MAX_SAME_TYPE:
                violations.append(
                    f"Event type {event_type} appears {count} times "
                    f"(max {self.MAX_SAME_TYPE})"
                )
        return ValidationResult(valid=len(violations) == 0, violations=violations)
