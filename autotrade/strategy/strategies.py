"""Concrete rule-based strategies.

These are starting points, not money printers. Each one is a few lines because
the point is that you can read it, understand exactly what it does, and then
change it. Add your own by subclassing Strategy and registering it below.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .base import Strategy


class SMACrossover(Strategy):
    """Classic trend following: long while the fast average is above the slow one.

    Works in trending markets, bleeds in choppy ones. `long_only=False` lets it
    flip short on the down-cross.
    """

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = True):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.long_only = fast, slow, long_only
        self.warmup = slow

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        fast = ind.sma(bars["close"], self.fast)
        slow = ind.sma(bars["close"], self.slow)
        raw = (fast > slow).astype(float)
        return raw if self.long_only else raw * 2 - 1


class RSIMeanReversion(Strategy):
    """Buy oversold, exit on recovery to the midline.

    Mean reversion is the opposite bet from trend following: it does well in
    ranges and gets run over in a sustained downtrend, which is exactly why the
    ATR stop in RiskConfig matters here.
    """

    def __init__(self, window: int = 14, entry: float = 30.0, exit: float = 55.0):
        self.window, self.entry, self.exit = window, entry, exit
        self.warmup = window * 3

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        rsi = ind.rsi(bars["close"], self.window)
        # State machine: enter when RSI crosses below `entry`, hold until it
        # crosses back above `exit`. Expressed vectorially via forward-fill.
        target = pd.Series(float("nan"), index=bars.index)
        target[rsi < self.entry] = 1.0
        target[rsi > self.exit] = 0.0
        return target.ffill().fillna(0.0)


class DonchianBreakout(Strategy):
    """Turtle-style breakout: long on a new N-bar high, out on an M-bar low."""

    def __init__(self, entry_window: int = 20, exit_window: int = 10):
        self.entry_window, self.exit_window = entry_window, exit_window
        self.warmup = max(entry_window, exit_window) + 1

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        # shift(1) so the channel is built from *prior* bars only.
        upper = bars["high"].rolling(self.entry_window).max().shift(1)
        lower = bars["low"].rolling(self.exit_window).min().shift(1)
        target = pd.Series(float("nan"), index=bars.index)
        target[bars["close"] > upper] = 1.0
        target[bars["close"] < lower] = 0.0
        return target.ffill().fillna(0.0)


class MACDTrend(Strategy):
    """Long while the MACD histogram is positive and price is above its 200 SMA.

    The long-term filter is the useful part: it keeps you from buying every
    bounce inside a bear market.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, trend_filter: int = 200):
        self.fast, self.slow, self.signal, self.trend_filter = fast, slow, signal, trend_filter
        self.warmup = max(slow + signal, trend_filter)

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        _, _, hist = ind.macd(bars["close"], self.fast, self.slow, self.signal)
        regime = bars["close"] > ind.sma(bars["close"], self.trend_filter)
        return ((hist > 0) & regime).astype(float)


class BuyAndHold(Strategy):
    """The benchmark you have to beat. Most strategies don't."""

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


class EMACrossover(Strategy):
    """Exponential Moving Average crossover.

    Weights recent prices more heavily than SMA, reacting faster to trend shifts.
    """

    def __init__(self, fast: int = 9, slow: int = 21, long_only: bool = True):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.long_only = fast, slow, long_only
        self.warmup = slow * 2

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        fast = ind.ema(bars["close"], self.fast)
        slow = ind.ema(bars["close"], self.slow)
        raw = (fast > slow).astype(float)
        return raw if self.long_only else raw * 2 - 1


class BollingerBandsStrategy(Strategy):
    """Mean reversion using Bollinger Bands.

    Buys when price closes below the lower band, exits when price reaches the
    middle moving average band.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window, self.num_std = window, num_std
        self.warmup = window + 5

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        _, mid, lower = ind.bollinger(bars["close"], self.window, self.num_std)
        target = pd.Series(float("nan"), index=bars.index)
        target[bars["close"] < lower] = 1.0
        target[bars["close"] >= mid] = 0.0
        return target.ffill().fillna(0.0)


class RSIStrategy(Strategy):
    """Standalone RSI momentum and reversion strategy.

    Enters long when RSI recovers above oversold threshold, exits when RSI
    reaches overbought threshold.
    """

    def __init__(
        self,
        window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        long_only: bool = True,
    ):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.long_only = long_only
        self.warmup = window * 3

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        rsi_series = ind.rsi(bars["close"], self.window)
        target = pd.Series(float("nan"), index=bars.index)
        target[rsi_series <= self.oversold] = 1.0
        target[rsi_series >= self.overbought] = 0.0 if self.long_only else -1.0
        return target.ffill().fillna(0.0)


REGISTRY: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossover,
    "ema_crossover": EMACrossover,
    "rsi_reversion": RSIMeanReversion,
    "rsi_strategy": RSIStrategy,
    "bollinger": BollingerBandsStrategy,
    "donchian": DonchianBreakout,
    "macd_trend": MACDTrend,
    "buy_and_hold": BuyAndHold,
}


def build(name: str, **kwargs) -> Strategy:
    """Instantiate a registered strategy by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)

