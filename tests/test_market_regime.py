"""Tests for Market Regime Detection Engine."""

import numpy as np
import pandas as pd
import pytest

from autotrade.market.regime import (
    RegimeResult,
    detect_regime,
    detect_market_regime_with_breadth,
)
from autotrade.market.constants import IST


def _create_synthetic_ohlcv(n_bars: int = 250, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B", tz=IST)
    
    # Generate geometric random walk with specified drift
    returns = rng.normal(trend, 0.015, size=n_bars)
    prices = 100.0 * np.cumprod(1 + returns)
    
    df = pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.002, n_bars)),
            "high": prices * (1 + np.abs(rng.normal(0.005, 0.003, n_bars))),
            "low": prices * (1 - np.abs(rng.normal(0.005, 0.003, n_bars))),
            "close": prices,
            "volume": rng.integers(100000, 500000, size=n_bars).astype(float),
        },
        index=dates,
    )
    return df


def test_regime_insufficient_data():
    df = _create_synthetic_ohlcv(n_bars=30)
    result = detect_regime(df, min_bars=60)
    assert not result.data_sufficient
    assert result.regime == "UNKNOWN"
    assert "Need at least 50 bars" in result.data_warning


def test_regime_bullish_detection():
    # Strong positive drift over 220 bars
    df = _create_synthetic_ohlcv(n_bars=220, trend=0.005, seed=123)
    result = detect_regime(df, min_bars=60)
    assert result.data_sufficient
    assert result.regime in ("BULLISH", "SIDEWAYS")
    assert 0.0 <= result.confidence <= 1.0
    assert result.explanation != ""


def test_regime_bearish_detection():
    # Strong negative drift
    df = _create_synthetic_ohlcv(n_bars=220, trend=-0.006, seed=456)
    result = detect_regime(df, min_bars=60)
    assert result.data_sufficient
    assert result.regime in ("BEARISH", "SIDEWAYS")
    assert 0.0 <= result.confidence <= 1.0


def test_market_breadth_and_universe_regime():
    data_cache = {
        "RELIANCE": _create_synthetic_ohlcv(200, trend=0.004, seed=1),
        "TCS": _create_synthetic_ohlcv(200, trend=0.003, seed=2),
        "INFY": _create_synthetic_ohlcv(200, trend=-0.003, seed=3),
    }
    regime_res = detect_market_regime_with_breadth(data_cache, reference_symbol="RELIANCE")
    
    assert regime_res.data_sufficient
    assert regime_res.regime in ("BULLISH", "BEARISH", "SIDEWAYS")
    assert 0.0 <= regime_res.breadth_pct <= 1.0
    assert "Market breadth" in regime_res.explanation
