"""Polymarket exchange interface.

Paper trading wrapper is mandatory — logs orders without executing for validation.
Real execution via py-clob-client deferred until paper trading is validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    PAPER = "PAPER"  # Paper trade — not submitted to exchange


@dataclass
class Order:
    contract_id: str
    side: OrderSide
    size: float            # In USDC
    price: float           # Limit price
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    fill_price: float | None = None
    order_id: str | None = None
    metadata: dict = field(default_factory=dict)


class PolymarketInterface:
    """Interface to Polymarket via py-clob-client.

    Starts in paper trading mode. Real execution requires explicit opt-in.
    """

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.order_log: list[Order] = []
        self._client = None  # py-clob-client instance — initialized on first real trade

    def get_market_price(self, contract_id: str) -> float | None:
        """Fetch current market price for a contract.

        Returns:
            Midpoint price in [0, 1], or None if unavailable.
        """
        # TODO: Implement with py-clob-client in Phase 5
        # For now, return None to signal no live data
        return None

    def place_order(
        self,
        contract_id: str,
        side: OrderSide,
        size: float,
        price: float,
    ) -> Order:
        """Place an order (paper or live).

        Args:
            contract_id: Polymarket condition ID.
            side: BUY or SELL.
            size: Position size in USDC.
            price: Limit price.

        Returns:
            Order record.
        """
        order = Order(
            contract_id=contract_id,
            side=side,
            size=size,
            price=price,
        )

        if self.paper_mode:
            order.status = OrderStatus.PAPER
            order.fill_price = price  # Assume fill at limit for paper
        else:
            order = self._submit_live_order(order)

        self.order_log.append(order)
        return order

    def _submit_live_order(self, order: Order) -> Order:
        """Submit order to Polymarket. Requires py-clob-client."""
        raise NotImplementedError(
            "Live trading not yet implemented. Use paper_mode=True."
        )

    def get_order_history(self) -> list[Order]:
        return list(self.order_log)

    def get_pnl_summary(self) -> dict:
        """Compute simple PnL summary from order log."""
        if not self.order_log:
            return {"total_orders": 0, "total_size": 0.0}

        total_size = sum(o.size for o in self.order_log)
        buys = [o for o in self.order_log if o.side == OrderSide.BUY]
        sells = [o for o in self.order_log if o.side == OrderSide.SELL]

        return {
            "total_orders": len(self.order_log),
            "total_size": total_size,
            "buys": len(buys),
            "sells": len(sells),
        }
