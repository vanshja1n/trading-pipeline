"""Live-loop sizing tests.

Motivated by a real defect: the backtester suppressed churn with
`rebalance_threshold`, the live loop did not, so a paper account produced
one-share round trips (sell 1 SPY, buy 1 SPY minutes later) that no backtest
would ever have shown. These tests pin the two paths together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrade.broker.base import Account, Broker, Order
from autotrade.config import LiveConfig, RiskConfig
from autotrade.engine import live as live_mod
from autotrade.engine.live import LiveTrader
from autotrade.risk.manager import RiskManager
from autotrade.strategy.strategies import BuyAndHold


class FakeBroker(Broker):
    def __init__(self, equity: float = 100_000.0, position: int = 0):
        self._equity = equity
        self._position = position
        self.orders: list[Order] = []

    def get_account(self) -> Account:
        return Account(equity=self._equity, cash=self._equity, buying_power=self._equity * 4)

    def get_position(self, symbol: str) -> int:
        return self._position

    def submit_order(self, symbol: str, qty: int, side: str) -> Order:
        order = Order(symbol=symbol, qty=qty, side=side, status="accepted")
        self.orders.append(order)
        return order

    def close_position(self, symbol: str):
        return None

    def is_market_open(self) -> bool:
        return True

    def cancel_all_orders(self) -> None:
        pass


class FakeFeed:
    def __init__(self, bars: pd.DataFrame):
        self.bars = bars

    def latest_bars(self, symbol, lookback, timeframe):
        return self.bars


def make_bars(price: float = 745.0, n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-07-31 09:30", periods=n, freq="min", tz="UTC")
    close = pd.Series(np.full(n, price), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
         "volume": 1e6},
        index=idx,
    )


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    """Never read or write the real bot's state file during tests."""
    monkeypatch.setattr(live_mod, "STATE_PATH", tmp_path / "live_state.json")


def build_trader(broker, bars, **cfg_kwargs) -> LiveTrader:
    config = LiveConfig(symbols=["SPY"], timeframe="1Min", **cfg_kwargs)
    return LiveTrader(
        strategy=BuyAndHold(),
        broker=broker,
        feed=FakeFeed(bars),
        config=config,
        risk_config=RiskConfig(max_position_weight=0.95),
    )


def test_one_share_drift_does_not_trade():
    """The exact bug: holding 127 when the target says 128 must not trade."""
    bars = make_bars(745.0)
    equity = 100_000.0
    target = RiskManager(RiskConfig(max_position_weight=0.95), equity).shares_for_weight(
        0.95, equity, 745.0
    )
    broker = FakeBroker(equity=equity, position=target - 1)

    build_trader(broker, bars)._handle_symbol("SPY", equity)

    assert broker.orders == [], f"churned: {broker.orders}"


def test_opening_from_flat_still_trades():
    bars = make_bars(745.0)
    broker = FakeBroker(equity=100_000.0, position=0)

    build_trader(broker, bars)._handle_symbol("SPY", 100_000.0)

    assert len(broker.orders) == 1
    assert broker.orders[0].side == "buy"


def test_exit_to_flat_is_never_suppressed():
    """Going to zero must always execute, however small the drift."""
    bars = make_bars(745.0)
    broker = FakeBroker(equity=100_000.0, position=1)
    trader = build_trader(broker, bars)
    trader.risk.halted = True  # forces target weight to 0

    trader._handle_symbol("SPY", 100_000.0)

    assert len(broker.orders) == 1
    assert broker.orders[0].side == "sell"
    assert broker.orders[0].qty == 1


def test_large_drift_rebalances():
    bars = make_bars(745.0)
    broker = FakeBroker(equity=100_000.0, position=40)  # far below a 0.95 target

    build_trader(broker, bars)._handle_symbol("SPY", 100_000.0)

    assert len(broker.orders) == 1
    assert broker.orders[0].side == "buy"


def test_threshold_of_zero_restores_old_behavior():
    bars = make_bars(745.0)
    equity = 100_000.0
    target = RiskManager(RiskConfig(max_position_weight=0.95), equity).shares_for_weight(
        0.95, equity, 745.0
    )
    broker = FakeBroker(equity=equity, position=target - 1)

    build_trader(broker, bars, rebalance_threshold=0.0)._handle_symbol("SPY", equity)

    assert len(broker.orders) == 1  # opt back in to churn if you really want it


def test_live_and_backtest_size_positions_identically():
    """Both paths must derive share counts from the same risk manager call."""
    risk_config = RiskConfig(max_position_weight=0.95)
    equity, price = 100_000.0, 745.0

    expected = RiskManager(risk_config, equity).shares_for_weight(0.95, equity, price)

    bars = make_bars(price)
    broker = FakeBroker(equity=equity, position=0)
    build_trader(broker, bars)._handle_symbol("SPY", equity)

    assert broker.orders[0].qty == expected


def test_equity_is_split_across_symbols():
    """Three symbols must each get a third of the account, not a full position."""
    bars = make_bars(745.0)
    equity = 100_000.0
    broker = FakeBroker(equity=equity, position=0)
    config = LiveConfig(symbols=["SPY", "QQQ", "GLD"], timeframe="1Min")
    trader = LiveTrader(BuyAndHold(), broker, FakeFeed(bars), config,
                        RiskConfig(max_position_weight=0.95))

    trader._handle_symbol("SPY", equity)

    one_third = RiskManager(RiskConfig(max_position_weight=0.95), equity).shares_for_weight(
        0.95, equity / 3, 745.0
    )
    assert broker.orders[0].qty == one_third
