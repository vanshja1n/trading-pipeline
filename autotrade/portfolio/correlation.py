"""Stock Correlation Analysis Engine.

Calculates pairwise correlations using actual historical daily returns from
the existing data cache. Provides:

1. Full correlation matrix
2. Annotated heatmap data
3. Highest and lowest correlated pairs
4. Interpretation text generated from actual calculated values

No fabricated correlations — if data is insufficient, reports that clearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CorrelationPair:
    """A pair of symbols with their correlation coefficient."""

    symbol_a: str
    symbol_b: str
    correlation: float

    def __str__(self) -> str:
        return f"{self.symbol_a} / {self.symbol_b}: {self.correlation:+.3f}"


@dataclass
class CorrelationResult:
    """Full correlation analysis output."""

    correlation_matrix: pd.DataFrame
    highest_pairs: list[CorrelationPair]
    lowest_pairs: list[CorrelationPair]
    interpretation: str
    symbols: list[str]
    data_points: int
    data_sufficient: bool = True
    warning: str = ""


def compute_correlation_analysis(
    data_cache: dict[str, pd.DataFrame],
    symbols: list[str],
    min_history: int = 60,
    top_n: int = 5,
) -> CorrelationResult:
    """Compute pairwise correlation matrix and identify extreme pairs.

    Parameters
    ----------
    data_cache : dict[str, pd.DataFrame]
        Symbol -> OHLCV DataFrame mapping from session state.
    symbols : list[str]
        Universe of symbols to analyze.
    min_history : int
        Minimum data points required per symbol.
    top_n : int
        Number of highest/lowest pairs to highlight.

    Returns
    -------
    CorrelationResult
    """
    # Collect close prices
    close_prices = {}
    for sym in symbols:
        df = data_cache.get(sym)
        if df is not None and not df.empty and len(df) >= min_history:
            close_prices[sym] = df["close"].copy()

    if len(close_prices) < 2:
        return CorrelationResult(
            correlation_matrix=pd.DataFrame(),
            highest_pairs=[],
            lowest_pairs=[],
            interpretation="Insufficient data for correlation analysis.",
            symbols=list(close_prices.keys()),
            data_points=0,
            data_sufficient=False,
            warning=(
                f"Need at least 2 symbols with {min_history}+ bars. "
                f"Only {len(close_prices)} symbol(s) qualified."
            ),
        )

    # Align to common dates and compute returns
    prices_df = pd.DataFrame(close_prices).dropna()

    if len(prices_df) < min_history:
        return CorrelationResult(
            correlation_matrix=pd.DataFrame(),
            highest_pairs=[],
            lowest_pairs=[],
            interpretation="Insufficient overlapping data for correlation analysis.",
            symbols=list(close_prices.keys()),
            data_points=len(prices_df),
            data_sufficient=False,
            warning=(
                f"After aligning dates, only {len(prices_df)} common data points. "
                f"Need at least {min_history}."
            ),
        )

    returns_df = prices_df.pct_change().dropna()
    valid_syms = list(returns_df.columns)
    n_points = len(returns_df)

    # Correlation matrix
    corr_matrix = returns_df.corr()

    # Extract unique pairs
    pairs: list[CorrelationPair] = []
    for i, sym_a in enumerate(valid_syms):
        for j, sym_b in enumerate(valid_syms):
            if i < j:
                corr_val = float(corr_matrix.loc[sym_a, sym_b])
                pairs.append(CorrelationPair(sym_a, sym_b, corr_val))

    # Sort for highest and lowest
    pairs_sorted = sorted(pairs, key=lambda p: p.correlation, reverse=True)
    highest = pairs_sorted[:top_n]
    lowest = pairs_sorted[-top_n:] if len(pairs_sorted) >= top_n else pairs_sorted[:]
    lowest = list(reversed(lowest))

    # Generate interpretation
    interpretation = _generate_interpretation(highest, lowest, valid_syms, corr_matrix)

    warning = ""
    if n_points < 120:
        warning = (
            f"Correlations computed from {n_points} data points. "
            "Results may be more stable with 120+ trading days."
        )

    return CorrelationResult(
        correlation_matrix=corr_matrix,
        highest_pairs=highest,
        lowest_pairs=lowest,
        interpretation=interpretation,
        symbols=valid_syms,
        data_points=n_points,
        data_sufficient=True,
        warning=warning,
    )


def _generate_interpretation(
    highest: list[CorrelationPair],
    lowest: list[CorrelationPair],
    symbols: list[str],
    corr_matrix: pd.DataFrame,
) -> str:
    """Generate human-readable interpretation from actual correlation values."""
    lines = []

    # Average correlation
    n = len(symbols)
    if n > 1:
        upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
        avg_corr = float(np.mean(upper_tri))
        lines.append(
            f"The average pairwise correlation across the {n}-stock universe is "
            f"{avg_corr:.3f}."
        )

        if avg_corr > 0.6:
            lines.append(
                "This suggests the portfolio is moderately concentrated — "
                "stocks tend to move together, reducing diversification benefit."
            )
        elif avg_corr > 0.3:
            lines.append(
                "This indicates moderate diversification — some stocks move together "
                "while others provide independent return streams."
            )
        else:
            lines.append(
                "This indicates strong diversification — stocks show relatively "
                "independent return behaviour."
            )

    # Highest correlated pair
    if highest:
        h = highest[0]
        lines.append(
            f"\n{h.symbol_a} and {h.symbol_b} show the highest correlation "
            f"({h.correlation:.3f}), suggesting they respond to similar market "
            f"drivers. Holding both provides limited diversification benefit."
        )

    # Lowest correlated pair
    if lowest:
        lo = lowest[0]
        if lo.correlation < 0.2:
            lines.append(
                f"\n{lo.symbol_a} and {lo.symbol_b} show the lowest correlation "
                f"({lo.correlation:.3f}), providing genuine diversification when "
                f"held together."
            )
        elif lo.correlation < 0:
            lines.append(
                f"\n{lo.symbol_a} and {lo.symbol_b} show negative correlation "
                f"({lo.correlation:.3f}), acting as natural hedges for each other."
            )

    # Sector-like groupings based on correlation clustering
    if len(highest) >= 2:
        high_syms = set()
        for p in highest[:3]:
            if p.correlation > 0.5:
                high_syms.add(p.symbol_a)
                high_syms.add(p.symbol_b)
        if len(high_syms) >= 3:
            lines.append(
                f"\nA cluster of highly correlated stocks ({', '.join(sorted(high_syms))}) "
                f"suggests sector concentration. Consider reducing overlap."
            )

    return " ".join(lines) if lines else "Correlation analysis completed."


# Alias for compatibility
compute_correlation_matrix = compute_correlation_analysis
