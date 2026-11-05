"""Tests for Portfolio Optimization Engine."""

import numpy as np
import pandas as pd
import pytest

from autotrade.portfolio.optimizer import (
    PortfolioAllocation,
    OptimizationResult,
    optimize_portfolio,
)
from autotrade.market.constants import IST


def _create_synthetic_ohlcv(symbol: str, n_bars: int = 150, mean_ret: float = 0.0005, vol: float = 0.015, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B", tz=IST)
    rets = rng.normal(mean_ret, vol, size=n_bars)
    prices = 1000.0 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 100000.0,
        },
        index=dates,
    )


def test_optimizer_insufficient_data():
    data_cache = {
        "RELIANCE": _create_synthetic_ohlcv("RELIANCE", 20, seed=1),
    }
    res = optimize_portfolio(data_cache, ["RELIANCE"], min_history=60)
    assert not res.data_sufficient
    assert "Need at least 2 symbols" in res.warning


def test_optimizer_equal_weight_and_methods():
    data_cache = {
        "RELIANCE": _create_synthetic_ohlcv("RELIANCE", 120, mean_ret=0.001, vol=0.012, seed=10),
        "TCS": _create_synthetic_ohlcv("TCS", 120, mean_ret=0.0008, vol=0.010, seed=20),
        "INFY": _create_synthetic_ohlcv("INFY", 120, mean_ret=0.0004, vol=0.018, seed=30),
    }
    symbols = ["RELIANCE", "TCS", "INFY"]
    res = optimize_portfolio(data_cache, symbols, min_history=60)
    
    assert res.data_sufficient
    assert len(res.optimized_allocations) >= 3

    # Check that Equal Weight allocation is present and weights sum to 1.0
    eq_alloc = next((a for a in res.optimized_allocations if a.name == "Equal Weight"), None)
    assert eq_alloc is not None
    assert len(eq_alloc.weights) == 3
    for sym in symbols:
        assert pytest.approx(eq_alloc.weights[sym], abs=1e-4) == 1.0 / 3.0
    assert pytest.approx(sum(eq_alloc.weights.values()), abs=1e-4) == 1.0

    # Verify all allocations have non-negative weights summing to 1.0
    for alloc in res.optimized_allocations:
        assert pytest.approx(sum(alloc.weights.values()), abs=1e-3) == 1.0
        assert all(w >= -1e-6 for w in alloc.weights.values())
        assert alloc.expected_volatility > 0.0
