"""Broker interface.

The engine only ever talks to this interface, which is what makes it possible
to run the exact same code path against a simulator, a paper account, or (once
you truly trust it) a real one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    invested_value: float = 0.0


@dataclass
class Order:
    symbol: str
    qty: int
    side: str  # "buy" | "sell"
    order_id: str | None = None
    status: str = "pending"
    filled_price: float | None = None


class Broker(ABC):
    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_position(self, symbol: str) -> int:
        """Signed share count; 0 when flat."""

    @abstractmethod
    def submit_order(self, symbol: str, qty: int, side: str) -> Order: ...

    @abstractmethod
    def close_position(self, symbol: str) -> Order | None: ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def cancel_all_orders(self) -> None: ...
