"""Portfolio Optimization Engine.

Implements mean-variance optimization (Markowitz) using actual historical returns
from the existing data cache. Supports:

1. Equal-weight baseline
2. Minimum variance portfolio
3. Maximum Sharpe ratio portfolio (tangency portfolio)
4. Risk parity (inverse-volatility weighting)

Uses only historical data available at the time of calculation — no look-ahead bias.
If insufficient data is available, reports that clearly rather than fabricating results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..strategy import indicators as ind


@dataclass
class PortfolioAllocation:
    """A single portfolio allocation with its expected metrics."""

    name: str  # e.g., "Current", "Max Sharpe", "Min Variance"
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float = 0.0
    data_sufficient: bool = True
    warning: str = ""


@dataclass
class OptimizationResult:
    """Complete optimization output."""

    current_allocation: PortfolioAllocation
    optimized_allocations: list[PortfolioAllocation]
    correlation_matrix: pd.DataFrame
    symbols: list[str]
    data_points: int
    data_sufficient: bool = True
    warning: str = ""
    risk_free_rate: float = 0.07  # Indian 10Y G-Sec


def _compute_returns_matrix(
    data_cache: dict[str, pd.DataFrame],
    symbols: list[str],
    min_history: int = 60,
) -> tuple[pd.DataFrame, list[str], str]:
    """Build aligned daily returns matrix from cached OHLCV data.

    Returns (returns_df, valid_symbols, warning_message).
    """
    close_prices = {}
    for sym in symbols:
        df = data_cache.get(sym)
        if df is not None and not df.empty and len(df) >= min_history:
            close_prices[sym] = df["close"].copy()

    if len(close_prices) < 2:
        return pd.DataFrame(), list(close_prices.keys()), (
            f"Need at least 2 symbols with {min_history}+ bars of data. "
            f"Only {len(close_prices)} symbol(s) qualified."
        )

    # Align all series to common dates
    prices_df = pd.DataFrame(close_prices)
    prices_df = prices_df.dropna()

    if len(prices_df) < min_history:
        return pd.DataFrame(), list(close_prices.keys()), (
            f"After aligning dates, only {len(prices_df)} common data points remain. "
            f"Need at least {min_history}."
        )

    returns_df = prices_df.pct_change().dropna()
    valid_syms = list(returns_df.columns)

    warning = ""
    if len(returns_df) < 120:
        warning = (
            f"Using {len(returns_df)} days of data. Results may be less stable "
            "with fewer than 120 trading days of history."
        )

    return returns_df, valid_syms, warning


def _portfolio_metrics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    periods_per_year: int = 248,
) -> tuple[float, float, float]:
    """Compute annualized return, volatility, and Sharpe for a given weight vector."""
    port_return = float(np.sum(mean_returns * weights) * periods_per_year)
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * periods_per_year, weights))))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Compute maximum drawdown from a return series."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0


def optimize_portfolio(
    data_cache: dict[str, pd.DataFrame],
    symbols: list[str],
    current_weights: dict[str, float] | None = None,
    risk_free_rate: float = 0.07,
    periods_per_year: int = 248,
    min_history: int = 60,
) -> OptimizationResult:
    """Run portfolio optimization on available market data.

    Parameters
    ----------
    data_cache : dict[str, pd.DataFrame]
        Symbol -> OHLCV DataFrame mapping.
    symbols : list[str]
        Universe of symbols to optimize across.
    current_weights : dict[str, float], optional
        Current portfolio weights. If None, assumes equal weight.
    risk_free_rate : float
        Risk-free rate for Sharpe calculation (default: Indian 10Y G-Sec ~7%).
    periods_per_year : int
        Trading days per year for annualization.
    min_history : int
        Minimum data points required.

    Returns
    -------
    OptimizationResult
    """
    returns_df, valid_syms, warning = _compute_returns_matrix(
        data_cache, symbols, min_history
    )

    if returns_df.empty:
        empty_alloc = PortfolioAllocation(
            name="Current",
            weights={s: 1.0 / len(symbols) for s in symbols},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            data_sufficient=False,
            warning=warning,
        )
        return OptimizationResult(
            current_allocation=empty_alloc,
            optimized_allocations=[],
            correlation_matrix=pd.DataFrame(),
            symbols=symbols,
            data_points=0,
            data_sufficient=False,
            warning=warning,
            risk_free_rate=risk_free_rate,
        )

    n_assets = len(valid_syms)
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    corr_matrix = returns_df.corr()

    # ── Current Allocation ────────────────────────────────────────────
    if current_weights:
        curr_w = np.array([current_weights.get(s, 0.0) for s in valid_syms])
        if curr_w.sum() > 0:
            curr_w = curr_w / curr_w.sum()
        else:
            curr_w = np.ones(n_assets) / n_assets
    else:
        curr_w = np.ones(n_assets) / n_assets

    curr_ret, curr_vol, curr_sharpe = _portfolio_metrics(
        curr_w, mean_returns, cov_matrix, risk_free_rate, periods_per_year
    )
    curr_port_returns = returns_df.dot(curr_w)
    curr_dd = _max_drawdown_from_returns(curr_port_returns)

    current_alloc = PortfolioAllocation(
        name="Current Portfolio",
        weights={s: float(w) for s, w in zip(valid_syms, curr_w)},
        expected_return=curr_ret,
        expected_volatility=curr_vol,
        sharpe_ratio=curr_sharpe,
        max_drawdown=curr_dd,
    )

    optimized_allocations: list[PortfolioAllocation] = []

    # ── 1. Equal Weight ───────────────────────────────────────────────
    eq_w = np.ones(n_assets) / n_assets
    eq_ret, eq_vol, eq_sharpe = _portfolio_metrics(
        eq_w, mean_returns, cov_matrix, risk_free_rate, periods_per_year
    )
    eq_port_returns = returns_df.dot(eq_w)
    eq_dd = _max_drawdown_from_returns(eq_port_returns)

    optimized_allocations.append(
        PortfolioAllocation(
            name="Equal Weight",
            weights={s: float(w) for s, w in zip(valid_syms, eq_w)},
            expected_return=eq_ret,
            expected_volatility=eq_vol,
            sharpe_ratio=eq_sharpe,
            max_drawdown=eq_dd,
        )
    )

    # ── 2. Minimum Variance Portfolio ─────────────────────────────────
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)
        min_var_w = inv_cov @ ones / (ones @ inv_cov @ ones)
        # Handle negative weights — constrain to long-only via projection
        min_var_w = np.maximum(min_var_w, 0.0)
        if min_var_w.sum() > 0:
            min_var_w = min_var_w / min_var_w.sum()
        else:
            min_var_w = eq_w.copy()

        mv_ret, mv_vol, mv_sharpe = _portfolio_metrics(
            min_var_w, mean_returns, cov_matrix, risk_free_rate, periods_per_year
        )
        mv_port_returns = returns_df.dot(min_var_w)
        mv_dd = _max_drawdown_from_returns(mv_port_returns)

        optimized_allocations.append(
            PortfolioAllocation(
                name="Minimum Variance",
                weights={s: float(w) for s, w in zip(valid_syms, min_var_w)},
                expected_return=mv_ret,
                expected_volatility=mv_vol,
                sharpe_ratio=mv_sharpe,
                max_drawdown=mv_dd,
            )
        )
    except np.linalg.LinAlgError:
        pass  # Singular covariance — skip this method

    # ── 3. Maximum Sharpe Ratio Portfolio ─────────────────────────────
    try:
        excess_returns = mean_returns - risk_free_rate / periods_per_year
        inv_cov = np.linalg.inv(cov_matrix)
        raw_w = inv_cov @ excess_returns
        # Constrain to long-only
        raw_w = np.maximum(raw_w, 0.0)
        if raw_w.sum() > 0:
            max_sharpe_w = raw_w / raw_w.sum()
        else:
            max_sharpe_w = eq_w.copy()

        ms_ret, ms_vol, ms_sharpe = _portfolio_metrics(
            max_sharpe_w, mean_returns, cov_matrix, risk_free_rate, periods_per_year
        )
        ms_port_returns = returns_df.dot(max_sharpe_w)
        ms_dd = _max_drawdown_from_returns(ms_port_returns)

        optimized_allocations.append(
            PortfolioAllocation(
                name="Max Sharpe Ratio",
                weights={s: float(w) for s, w in zip(valid_syms, max_sharpe_w)},
                expected_return=ms_ret,
                expected_volatility=ms_vol,
                sharpe_ratio=ms_sharpe,
                max_drawdown=ms_dd,
            )
        )
    except np.linalg.LinAlgError:
        pass

    # ── 4. Risk Parity (Inverse-Volatility) ───────────────────────────
    vols = np.sqrt(np.diag(cov_matrix))
    if np.all(vols > 0):
        inv_vol_w = (1.0 / vols) / (1.0 / vols).sum()
        rp_ret, rp_vol, rp_sharpe = _portfolio_metrics(
            inv_vol_w, mean_returns, cov_matrix, risk_free_rate, periods_per_year
        )
        rp_port_returns = returns_df.dot(inv_vol_w)
        rp_dd = _max_drawdown_from_returns(rp_port_returns)

        optimized_allocations.append(
            PortfolioAllocation(
                name="Risk Parity",
                weights={s: float(w) for s, w in zip(valid_syms, inv_vol_w)},
                expected_return=rp_ret,
                expected_volatility=rp_vol,
                sharpe_ratio=rp_sharpe,
                max_drawdown=rp_dd,
            )
        )

    return OptimizationResult(
        current_allocation=current_alloc,
        optimized_allocations=optimized_allocations,
        correlation_matrix=corr_matrix,
        symbols=valid_syms,
        data_points=len(returns_df),
        data_sufficient=True,
        warning=warning,
        risk_free_rate=risk_free_rate,
    )
