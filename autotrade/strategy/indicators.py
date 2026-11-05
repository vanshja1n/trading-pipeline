"""Vectorized technical indicators.

Every function returns a Series aligned to the input index. None of them peek
at future bars -- that property is what makes the backtest honest, so keep it
that way if you add more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # A window with zero losses is a maximally overbought reading.
    return out.where(avg_loss != 0, 100.0).where(avg_gain != 0, out)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (lower, mid, upper)."""
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    return mid - num_std * std, mid, mid + num_std * std


def realized_volatility(close: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling standard deviation of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)
