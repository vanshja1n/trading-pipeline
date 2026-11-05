"""Tests for the pieces where a bug costs real money."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrade.config import BacktestConfig, RiskConfig
from autotrade.engine.backtest import Backtester
from autotrade.reporting import metrics as metrics_mod
from autotrade.risk.manager import RiskManager
from autotrade.strategy import indicators as ind
from autotrade.strategy.base import Strategy
from autotrade.strategy.strategies import BuyAndHold, SMACrossover


def make_bars(closes: list[float], spread: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D", tz="UTC")
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def trending_bars(n: int = 300, drift: float = 0.0005, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.01, n)
    closes = 100 * np.exp(np.cumsum(returns))
    return make_bars(list(closes))


# --------------------------------------------------------------- indicators


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).iloc[-1] == pytest.approx(4.0)
    assert pd.isna(ind.sma(s, 3).iloc[1])  # no value before warmup


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(10, 50, 100))
    down = pd.Series(np.linspace(50, 10, 100))
    assert ind.rsi(up).iloc[-1] > 70
    assert ind.rsi(down).iloc[-1] < 30
    assert ind.rsi(up).dropna().between(0, 100).all()


def test_atr_is_positive():
    bars = trending_bars(120)
    atr = ind.atr(bars["high"], bars["low"], bars["close"]).dropna()
    assert len(atr) > 0 and (atr > 0).all()


# ------------------------------------------------------------------- no lookahead


def test_signals_do_not_use_future_bars():
    """The single most important test in the repo.

    Truncating the data must not change any signal already produced. If it
    does, the strategy is peeking at the future and the backtest is fiction.
    """
    bars = trending_bars(300)
    for strategy in (SMACrossover(10, 30), BuyAndHold()):
        full = strategy(bars)
        partial = strategy(bars.iloc[:200])
        pd.testing.assert_series_equal(full.iloc[:200], partial, check_freq=False)


def test_engine_executes_at_next_open_not_current_close():
    """A signal that fires on bar t must not be filled before bar t+1's open."""

    class FireOnBarFive(Strategy):
        warmup = 0

        def generate_signals(self, bars):
            s = pd.Series(0.0, index=bars.index)
            s.iloc[5:] = 1.0
            return s

    bars = make_bars([100.0] * 10)
    result = Backtester(FireOnBarFive(), BacktestConfig(slippage_bps=0)).run(bars, "TEST")
    fills = result.trades[result.trades["reason"] == "signal"]
    assert not fills.empty
    assert fills.iloc[0]["timestamp"] == bars.index[6]


# ------------------------------------------------------------------- backtester


def test_buy_and_hold_tracks_the_asset():
    bars = trending_bars(400, drift=0.001)
    cfg = BacktestConfig(starting_cash=100_000, slippage_bps=0)
    risk = RiskConfig(max_position_weight=1.0, max_drawdown=0.99)
    result = Backtester(BuyAndHold(), cfg, risk).run(bars, "TEST")

    asset_return = bars["close"].iloc[-1] / bars["open"].iloc[1] - 1
    # Whole-share rounding leaves a little idle cash, so allow a small gap.
    assert result.metrics.total_return == pytest.approx(asset_return, abs=0.02)


def test_costs_reduce_returns():
    bars = trending_bars(400, seed=3)
    strategy = SMACrossover(5, 20)
    free = Backtester(strategy, BacktestConfig(slippage_bps=0)).run(bars, "T")
    costly = Backtester(strategy, BacktestConfig(slippage_bps=50)).run(bars, "T")
    assert costly.metrics.total_return < free.metrics.total_return


def test_equity_never_goes_negative_on_long_only():
    bars = trending_bars(300, drift=-0.003, seed=11)
    result = Backtester(SMACrossover(5, 20), BacktestConfig()).run(bars, "T")
    assert (result.equity > 0).all()


def test_no_trades_means_flat_equity():
    class NeverTrade(Strategy):
        def generate_signals(self, bars):
            return pd.Series(0.0, index=bars.index)

    bars = trending_bars(100)
    result = Backtester(NeverTrade(), BacktestConfig(starting_cash=5_000)).run(bars, "T")
    assert result.equity.nunique() == 1
    assert result.metrics.final_equity == pytest.approx(5_000)


# ------------------------------------------------------------------------ risk


def test_kill_switch_halts_and_flattens():
    risk = RiskManager(RiskConfig(max_drawdown=0.10), 1000.0)
    assert not risk.update_equity(1100.0)
    assert not risk.update_equity(1050.0)
    assert risk.update_equity(950.0)  # -13.6% from the 1100 peak
    assert risk.target_weight(1.0) == 0.0


def test_kill_switch_fires_in_a_backtest():
    bars = make_bars(list(np.linspace(100, 40, 200)))  # steady 60% decline
    risk = RiskConfig(max_drawdown=0.10, max_position_weight=1.0)
    result = Backtester(BuyAndHold(), BacktestConfig(), risk).run(bars, "T")
    assert result.halted
    assert result.metrics.max_drawdown < 0.35  # far better than holding to -60%
    assert result.positions.iloc[-1] == 0


def test_daily_loss_limit():
    risk = RiskManager(RiskConfig(max_daily_loss=0.03, max_drawdown=0.99), 10_000.0)
    risk.start_session(10_000.0)
    assert not risk.update_equity(9_800.0)
    assert risk.update_equity(9_600.0)


def test_position_sizing_respects_max_weight():
    risk = RiskManager(RiskConfig(max_position_weight=0.5), 10_000.0)
    assert risk.target_weight(1.0) == pytest.approx(0.5)
    assert risk.shares_for_weight(0.5, 10_000, 100.0) == 50


def test_vol_targeting_shrinks_position_in_wild_markets():
    risk = RiskManager(RiskConfig(target_volatility=0.15, max_position_weight=1.0), 10_000.0)
    calm = risk.target_weight(1.0, volatility=0.10)
    wild = risk.target_weight(1.0, volatility=0.60)
    assert wild < calm
    assert calm <= 1.0  # max_leverage caps the scale-up


def test_atr_stop_exits_the_position():
    # Steady climb, then a cliff that must trip a 2-ATR stop.
    closes = [100 + i * 0.5 for i in range(60)] + [110, 95, 90, 88, 87]
    bars = make_bars(closes)
    # Disable the other two protections so this test isolates the ATR stop.
    risk = RiskConfig(stop_loss_atr=2.0, max_drawdown=0.99, max_daily_loss=0.99)
    result = Backtester(BuyAndHold(), BacktestConfig(), risk).run(bars, "T")
    assert result.trades["reason"].str.contains("stop").any()


# --------------------------------------------------------------------- metrics


def test_metrics_on_a_known_curve():
    idx = pd.date_range("2020-01-01", periods=253, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(10_000, 12_000, 253), index=idx)
    m = metrics_mod.compute(equity, periods_per_year=252)
    assert m.total_return == pytest.approx(0.20)
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)
    assert m.sharpe > 0


def test_max_drawdown_is_measured_peak_to_trough():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    equity = pd.Series([100.0, 120, 60, 80, 110], index=idx)
    assert metrics_mod.compute(equity).max_drawdown == pytest.approx(0.5)


def test_win_rate_counts_closed_trades_only():
    trades = pd.DataFrame({"pnl": [10.0, -5.0, None, 20.0]})
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    m = metrics_mod.compute(pd.Series([100.0, 101, 102, 103], index=idx), trades)
    assert m.num_trades == 3
    assert m.win_rate == pytest.approx(2 / 3)


# ------------------------------------------------------------------ validation


def test_bad_config_is_rejected_early():
    with pytest.raises(ValueError):
        RiskConfig(max_position_weight=1.5)
    with pytest.raises(ValueError):
        RiskConfig(max_drawdown=0.0)
    with pytest.raises(ValueError):
        SMACrossover(fast=50, slow=20)


def test_missing_columns_raise():
    bars = trending_bars(100).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        BuyAndHold()(bars)
