"""Kalshi exchange interface — deferred until Polymarket works.

Stub implementation with the same interface pattern as PolymarketInterface.
"""

from __future__ import annotations

from radiant_seer.execution.poly_interface import Order, OrderSide, OrderStatus


class KalshiInterface:
    """Interface to Kalshi REST API via httpx. Deferred implementation."""

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.order_log: list[Order] = []

    def get_market_price(self, ticker: str) -> float | None:
        """Fetch current market price for a Kalshi contract."""
        # TODO: Implement with httpx in Phase 5+
        return None

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        size: float,
        price: float,
    ) -> Order:
        """Place an order (paper only for now)."""
        order = Order(
            contract_id=ticker,
            side=side,
            size=size,
            price=price,
            status=OrderStatus.PAPER,
            fill_price=price,
        )
        self.order_log.append(order)
        return order
