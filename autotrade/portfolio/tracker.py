"""Portfolio management and multi-asset position tracking module.

Provides real-time portfolio valuation, sector/asset allocation breakdowns,
and equity curve tracking in INR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import pandas as pd

from ..market.constants import IST


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    invested_equity: float
    total_equity: float
    allocations: dict[str, float] = field(default_factory=dict)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class PortfolioTracker:
    """Manages multi-stock positions and portfolio metrics."""

    def __init__(self, starting_cash: float = 500_000.0):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.positions: dict[str, int] = {}
        self.avg_entry_prices: dict[str, float] = {}
        self.current_prices: dict[str, float] = {}
        self.realized_pnl: float = 0.0
        self.history: list[PortfolioSnapshot] = []

    def update_price(self, symbol: str, price: float) -> None:
        """Update market price for an asset."""
        self.current_prices[symbol.upper()] = float(price)

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        fill_price: float,
        costs: float = 0.0,
    ) -> None:
        """Update internal positions after an order execution."""
        sym = symbol.upper()
        side_lower = side.lower()
        delta = qty if side_lower == "buy" else -qty
        current_qty = self.positions.get(sym, 0)
        new_qty = current_qty + delta

        # P&L on closing portion
        if current_qty != 0 and (current_qty > 0) != (delta > 0):
            closed = min(abs(delta), abs(current_qty))
            entry_p = self.avg_entry_prices.get(sym, fill_price)
            realized = (fill_price - entry_p) * closed * (1 if current_qty > 0 else -1)
            self.realized_pnl += realized

        # Cash effect
        self.cash -= delta * fill_price + costs

        if new_qty == 0:
            self.positions.pop(sym, None)
            self.avg_entry_prices.pop(sym, None)
        elif current_qty == 0 or (current_qty > 0) == (delta > 0):
            old_qty = abs(current_qty)
            old_avg = self.avg_entry_prices.get(sym, 0.0)
            new_avg = (old_qty * old_avg + qty * fill_price) / abs(new_qty)
            self.positions[sym] = new_qty
            self.avg_entry_prices[sym] = new_avg
        else:
            self.positions[sym] = new_qty

        self.update_price(sym, fill_price)

    def get_snapshot(self) -> PortfolioSnapshot:
        """Calculate current portfolio valuation and weights."""
        invested = 0.0
        unrealized = 0.0
        allocations: dict[str, float] = {}

        for sym, shares in self.positions.items():
            price = self.current_prices.get(sym, self.avg_entry_prices.get(sym, 0.0))
            val = shares * price
            invested += val
            entry = self.avg_entry_prices.get(sym, price)
            unrealized += (price - entry) * shares
            allocations[sym] = val

        total_equity = self.cash + invested
        # Normalize weights
        weights = {}
        if total_equity > 0:
            for sym, val in allocations.items():
                weights[sym] = val / total_equity
            weights["CASH"] = self.cash / total_equity

        snap = PortfolioSnapshot(
            timestamp=datetime.now(IST),
            cash=self.cash,
            invested_equity=invested,
            total_equity=total_equity,
            allocations=weights,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
        )
        self.history.append(snap)
        return snap

    def get_equity_series(self) -> pd.Series:
        """Return historical equity curve as a pandas Series."""
        if not self.history:
            return pd.Series([self.starting_cash])
        times = [s.timestamp for s in self.history]
        values = [s.total_equity for s in self.history]
        return pd.Series(values, index=pd.DatetimeIndex(times, tz=IST), name="equity")
