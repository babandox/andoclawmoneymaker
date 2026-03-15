"""Kelly Criterion position sizing.

f* = (p * b - q) / b

Quarter-Kelly (0.25x) by default — full Kelly is too aggressive given estimation error.
Hard cap at max_risk per event to prevent ruin.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionSize:
    fraction: float       # Fraction of bankroll to risk
    direction: str        # "BUY" or "SELL"
    edge: float           # Expected edge (p_model - p_market)
    confidence: float     # Model confidence
    capped: bool          # Whether the position was capped by max_risk


class KellySizer:
    """Compute position sizes using fractional Kelly criterion."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_risk: float = 0.02,
        min_edge: float = 0.05,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_risk = max_risk
        self.min_edge = min_edge

    def compute(
        self,
        p_model: float,
        p_market: float,
        confidence: float = 1.0,
    ) -> PositionSize:
        """Compute Kelly-optimal position size.

        Args:
            p_model: Model's estimated probability of YES outcome.
            p_market: Market price (probability implied by market).
            confidence: Model confidence multiplier in [0, 1].

        Returns:
            PositionSize with sizing details.
        """
        edge = p_model - p_market
        direction = "BUY" if edge > 0 else "SELL"
        abs_edge = abs(edge)

        # No position if edge below minimum
        if abs_edge < self.min_edge:
            return PositionSize(
                fraction=0.0, direction=direction, edge=edge,
                confidence=confidence, capped=False,
            )

        # Kelly formula for binary outcomes
        # For BUY: p = p_model, b = (1/p_market - 1) = odds offered by market
        # For SELL: p = 1 - p_model, b = (1/(1-p_market) - 1)
        if direction == "BUY":
            p = p_model
            q = 1 - p
            b = (1.0 / max(p_market, 0.01)) - 1.0
        else:
            p = 1 - p_model
            q = p_model
            b = (1.0 / max(1 - p_market, 0.01)) - 1.0

        if b <= 0:
            return PositionSize(
                fraction=0.0, direction=direction, edge=edge,
                confidence=confidence, capped=False,
            )

        # Kelly fraction: f* = (p*b - q) / b
        f_star = (p * b - q) / b
        f_star = max(0.0, f_star)

        # Apply fractional Kelly and confidence scaling
        f_adjusted = f_star * self.kelly_fraction * confidence

        # Cap at max risk
        capped = f_adjusted > self.max_risk
        f_final = min(f_adjusted, self.max_risk)

        return PositionSize(
            fraction=f_final,
            direction=direction,
            edge=edge,
            confidence=confidence,
            capped=capped,
        )
