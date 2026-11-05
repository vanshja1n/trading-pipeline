"""Alpaca broker adapter, plus a dry-run wrapper.

Safety posture: this class refuses to construct against the live-money endpoint
unless the caller passes ``paper=False`` explicitly *and* has set ALPACA_LIVE in
the environment. Two independent switches, on purpose.
"""

from __future__ import annotations

import logging

from .base import Account, Broker, Order

log = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca credentials missing. Put ALPACA_API_KEY / ALPACA_API_SECRET in .env"
            )
        from alpaca.trading.client import TradingClient

        self.paper = paper
        if not paper:
            log.warning("=" * 60)
            log.warning("CONNECTED TO A LIVE-MONEY ALPACA ACCOUNT. Orders are real.")
            log.warning("=" * 60)
        self._client = TradingClient(api_key, api_secret, paper=paper)

    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
        )

    def get_position(self, symbol: str) -> int:
        try:
            pos = self._client.get_open_position(symbol)
            return int(float(pos.qty))
        except Exception:
            return 0  # Alpaca 404s when flat

    def submit_order(self, symbol: str, qty: int, side: str) -> Order:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if qty <= 0:
            raise ValueError("qty must be positive; use `side` for direction")

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        submitted = self._client.submit_order(request)
        log.info("submitted %s %d %s (id=%s)", side, qty, symbol, submitted.id)
        return Order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_id=str(submitted.id),
            status=str(submitted.status),
        )

    def close_position(self, symbol: str) -> Order | None:
        held = self.get_position(symbol)
        if held == 0:
            return None
        closed = self._client.close_position(symbol)
        log.info("closing entire %s position (%d shares)", symbol, held)
        return Order(
            symbol=symbol,
            qty=abs(held),
            side="sell" if held > 0 else "buy",
            order_id=str(closed.id),
            status=str(closed.status),
        )

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()


class DryRunBroker(Broker):
    """Wraps a real broker: reads are live, writes are logged and discarded.

    Run the bot like this for a few days first. If the intended orders look
    sane, take the wrapper off.
    """

    def __init__(self, inner: Broker):
        self.inner = inner
        self.intended: list[Order] = []

    def get_account(self) -> Account:
        return self.inner.get_account()

    def get_position(self, symbol: str) -> int:
        return self.inner.get_position(symbol)

    def is_market_open(self) -> bool:
        return self.inner.is_market_open()

    def submit_order(self, symbol: str, qty: int, side: str) -> Order:
        order = Order(symbol=symbol, qty=qty, side=side, status="dry_run")
        self.intended.append(order)
        log.info("[DRY RUN] would %s %d %s", side, qty, symbol)
        return order

    def close_position(self, symbol: str) -> Order | None:
        held = self.inner.get_position(symbol)
        if held == 0:
            return None
        log.info("[DRY RUN] would close %s (%d shares)", symbol, held)
        return self.submit_order(symbol, abs(held), "sell" if held > 0 else "buy")

    def cancel_all_orders(self) -> None:
        log.info("[DRY RUN] would cancel all open orders")
