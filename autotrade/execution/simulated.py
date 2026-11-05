"""Simulated broker for Indian equity paper trading.

Maintains cash, whole-share positions, realized/unrealized P&L, and order fills
locally with realistic Indian transaction costs. Operates without any external
broker API dependency.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..broker.base import Account, Broker, Order
from ..market.calendar import NSECalendar
from ..market.constants import IST
from .costs import CostBreakdown, IndianEquityCostModel, TransactionCostModel

log = logging.getLogger(__name__)


@dataclass
class PositionDetail:
    symbol: str
    shares: int
    avg_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


class SimulatedBroker(Broker):
    """Local simulated broker for paper trading Indian equities.

    Tracks INR balances, position lots, trade fills, and SEBI/statutory transaction costs.
    """

    def __init__(
        self,
        starting_cash: float = 500_000.0,  # ₹5,00,000 INR
        cost_model: TransactionCostModel | None = None,
        slippage_bps: float = 2.0,
    ):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.cost_model = cost_model or IndianEquityCostModel()
        self.slippage_bps = slippage_bps
        self.calendar = NSECalendar()

        self._positions: dict[str, int] = {}
        self._avg_prices: dict[str, float] = {}
        self._current_prices: dict[str, float] = {}
        self._orders: list[Order] = []
        self._trade_history: list[dict] = []
        self._realized_pnl: float = 0.0
        self._kill_switch_engaged: bool = False

    def update_price(self, symbol: str, price: float) -> None:
        """Update latest known market price for a symbol."""
        sym = symbol.upper()
        self._current_prices[sym] = float(price)

    def get_position(self, symbol: str) -> int:
        """Signed share count for symbol."""
        return self._positions.get(symbol.upper(), 0)

    def get_position_detail(self, symbol: str) -> PositionDetail | None:
        sym = symbol.upper()
        shares = self.get_position(sym)
        if shares == 0:
            return None
        avg_p = self._avg_prices.get(sym, 0.0)
        curr_p = self._current_prices.get(sym, avg_p)
        return PositionDetail(symbol=sym, shares=shares, avg_price=avg_p, current_price=curr_p)

    def get_all_positions(self) -> list[PositionDetail]:
        details = []
        for sym, shares in self._positions.items():
            if shares != 0:
                detail = self.get_position_detail(sym)
                if detail:
                    details.append(detail)
        return details

    def get_account(self) -> Account:
        """Compute current account value in INR.

        Returns a fully reconciled Account with:
        - equity = cash + sum(position market values)
        - unrealized_pnl = sum of per-position unrealized P&L
        - realized_pnl = cumulative realized P&L from closed trades
        - invested_value = sum of position market values
        """
        portfolio_val = 0.0
        unrealized_pnl = 0.0
        for sym, shares in self._positions.items():
            price = self._current_prices.get(sym, self._avg_prices.get(sym, 0.0))
            mkt_val = shares * price
            portfolio_val += mkt_val
            entry = self._avg_prices.get(sym, price)
            unrealized_pnl += (price - entry) * shares

        equity = self.cash + portfolio_val
        return Account(
            equity=equity,
            cash=self.cash,
            buying_power=max(0.0, self.cash),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            invested_value=portfolio_val,
        )

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        price: float | None = None,
    ) -> Order:
        """Execute a market or specified-price order locally with transaction costs."""
        if getattr(self, "_kill_switch_engaged", False):
            raise RuntimeError("Order rejected: Emergency Kill Switch is engaged. All trading is halted.")

        sym = symbol.upper()
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        raw_price = price or self._current_prices.get(sym)
        if raw_price is None or raw_price <= 0:
            raise ValueError(f"no valid market price available to execute {sym}")

        # Apply slippage
        slip_pct = self.slippage_bps / 10_000.0
        fill_price = raw_price * (1.0 + slip_pct) if side_lower == "buy" else raw_price * (1.0 - slip_pct)

        # Compute Indian statutory & exchange costs
        costs = self.cost_model.calculate(side_lower, qty, fill_price)
        total_costs = costs.total

        current_pos = self._positions.get(sym, 0)
        delta = qty if side_lower == "buy" else -qty
        new_pos = current_pos + delta

        # Realized P&L check on closing or reduction
        realized = 0.0
        if current_pos != 0 and (current_pos > 0) != (delta > 0):
            closed_shares = min(abs(delta), abs(current_pos))
            entry_p = self._avg_prices.get(sym, fill_price)
            realized = (fill_price - entry_p) * closed_shares * (1 if current_pos > 0 else -1)
            self._realized_pnl += realized

        # Cash adjustment
        cash_delta = delta * fill_price + total_costs
        self.cash -= cash_delta

        # Position updates
        if new_pos == 0:
            self._positions.pop(sym, None)
            self._avg_prices.pop(sym, None)
        elif current_pos == 0 or (current_pos > 0) == (delta > 0):
            old_qty = abs(current_pos)
            old_avg = self._avg_prices.get(sym, 0.0)
            new_avg = (old_qty * old_avg + qty * fill_price) / abs(new_pos)
            self._positions[sym] = new_pos
            self._avg_prices[sym] = new_avg
        else:
            self._positions[sym] = new_pos

        order_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            symbol=sym,
            qty=qty,
            side=side_lower,
            order_id=order_id,
            status="filled",
            filled_price=fill_price,
        )
        self._orders.append(order)

        self._trade_history.append(
            {
                "order_id": order_id,
                "timestamp": datetime.now(IST).isoformat(),
                "symbol": sym,
                "side": side_lower,
                "qty": qty,
                "price": fill_price,
                "costs": total_costs,
                "realized_pnl": realized,
                "position_after": new_pos,
            }
        )

        return order

    def close_position(self, symbol: str) -> Order | None:
        """Close any open position for the given symbol."""
        pos = self.get_position(symbol)
        if pos == 0:
            return None
        side = "sell" if pos > 0 else "buy"
        return self.submit_order(symbol, abs(pos), side)

    def is_market_open(self) -> bool:
        """Check if NSE equity segment is open right now in IST."""
        return self.calendar.is_market_open()

    def cancel_all_orders(self) -> None:
        """No-op for simulated market fills."""
        pass

    def engage_kill_switch(self) -> int:
        """Halt all trading and close all open positions immediately."""
        closed = 0
        for sym in list(self._positions.keys()):
            try:
                self.close_position(sym)
                closed += 1
            except Exception as e:
                log.error("Failed to close %s on kill switch: %s", sym, e)
        self._kill_switch_engaged = True
        return closed

    def reset_kill_switch(self) -> None:
        """Re-enable trading."""
        self._kill_switch_engaged = False
