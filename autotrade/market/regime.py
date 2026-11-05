"""Market Regime Detection Engine.

Classifies the current market environment into BULLISH, BEARISH, or SIDEWAYS
using quantitative signals computed from existing market data:

1. Trend direction (price vs. moving averages)
2. Moving-average relationships (fast vs. slow SMA/EMA)
3. Momentum (RSI, MACD histogram)
4. Volatility regime (realized vol vs. historical norms)
5. Market breadth (fraction of watchlist stocks above their own MAs)
6. Recent market returns (20-day, 50-day cumulative)

All calculations use the existing indicator library and avoid look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..strategy import indicators as ind


@dataclass
class RegimeResult:
    """Output of market regime classification."""

    regime: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    confidence: float  # 0.0 to 1.0
    trend_label: str  # "Strong", "Moderate", "Weak", "Neutral"
    volatility_label: str  # "Low", "Moderate", "High", "Extreme"
    explanation: str  # Human-readable reasoning

    # Underlying signal components for transparency
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    breadth_score: float = 0.0
    returns_score: float = 0.0

    # Raw values
    current_price: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    rsi_14: float = 0.0
    realized_vol_20: float = 0.0
    recent_return_20d: float = 0.0
    breadth_pct: float = 0.0  # % of stocks above their 50-day SMA

    data_sufficient: bool = True
    data_warning: str = ""


def detect_regime(
    df: pd.DataFrame,
    min_bars: int = 200,
) -> RegimeResult:
    """Detect market regime for a single symbol's OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Must have columns:
        open, high, low, close, volume.
    min_bars : int
        Minimum bars required for reliable classification.

    Returns
    -------
    RegimeResult
    """
    if df is None or df.empty or len(df) < 50:
        return RegimeResult(
            regime="UNKNOWN",
            confidence=0.0,
            trend_label="Insufficient Data",
            volatility_label="Insufficient Data",
            explanation="Not enough historical data to classify the market regime.",
            data_sufficient=False,
            data_warning=f"Need at least 50 bars, got {len(df) if df is not None else 0}.",
        )

    close = df["close"]
    n = len(close)
    has_full_history = n >= min_bars

    # ── Component 1: Trend Direction ──────────────────────────────────
    # Price vs. 50-day and 200-day SMA
    sma_50 = ind.sma(close, 50)
    sma_200 = ind.sma(close, min(200, n - 1)) if n > 200 else ind.sma(close, max(50, n - 1))
    ema_20 = ind.ema(close, 20)

    current_price = float(close.iloc[-1])
    sma_50_val = float(sma_50.iloc[-1]) if pd.notna(sma_50.iloc[-1]) else current_price
    sma_200_val = float(sma_200.iloc[-1]) if pd.notna(sma_200.iloc[-1]) else current_price

    trend_score = 0.0
    # Price above 50 SMA: +1, below: -1
    if sma_50_val > 0:
        trend_score += 1.0 if current_price > sma_50_val else -1.0
    # Price above 200 SMA: +1, below: -1
    if sma_200_val > 0 and has_full_history:
        trend_score += 1.0 if current_price > sma_200_val else -1.0
    # 50 SMA above 200 SMA (golden cross vs. death cross): +1 / -1
    if sma_50_val > 0 and sma_200_val > 0 and has_full_history:
        trend_score += 0.5 if sma_50_val > sma_200_val else -0.5
    # EMA 20 slope (positive recent trend)
    if len(ema_20.dropna()) >= 5:
        ema_slope = float(ema_20.iloc[-1] - ema_20.iloc[-5])
        trend_score += 0.5 if ema_slope > 0 else -0.5

    # Normalize to [-1, 1]
    max_trend = 3.0 if has_full_history else 1.5
    trend_score = max(-1.0, min(1.0, trend_score / max_trend))

    # ── Component 2: Momentum ─────────────────────────────────────────
    rsi_val = float(ind.rsi(close, 14).iloc[-1]) if n > 14 else 50.0
    _, _, macd_hist = ind.macd(close)
    macd_val = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0

    momentum_score = 0.0
    # RSI interpretation
    if rsi_val > 60:
        momentum_score += (rsi_val - 50) / 50.0  # max ~1.0 at RSI=100
    elif rsi_val < 40:
        momentum_score -= (50 - rsi_val) / 50.0  # min ~-1.0 at RSI=0
    # MACD histogram sign
    momentum_score += 0.5 if macd_val > 0 else -0.5
    momentum_score = max(-1.0, min(1.0, momentum_score))

    # ── Component 3: Volatility ───────────────────────────────────────
    vol_20 = float(ind.realized_volatility(close, 20, 248).iloc[-1]) if n > 20 else 0.0
    vol_60 = float(ind.realized_volatility(close, min(60, n - 1), 248).iloc[-1]) if n > 60 else vol_20

    # High vol => more uncertainty, biases toward SIDEWAYS or BEARISH
    volatility_score = 0.0
    if vol_20 > 0.35:
        volatility_score = -0.8  # Extreme volatility
    elif vol_20 > 0.25:
        volatility_score = -0.4  # High volatility
    elif vol_20 > 0.15:
        volatility_score = 0.0  # Moderate
    else:
        volatility_score = 0.3  # Low vol, often bullish/trending

    # Rising vol vs. falling vol
    if vol_60 > 0:
        vol_ratio = vol_20 / vol_60
        if vol_ratio > 1.3:
            volatility_score -= 0.2  # Vol expanding
        elif vol_ratio < 0.7:
            volatility_score += 0.2  # Vol contracting

    volatility_score = max(-1.0, min(1.0, volatility_score))

    # ── Component 4: Recent Returns ───────────────────────────────────
    ret_20d = float(close.iloc[-1] / close.iloc[-min(20, n - 1)] - 1) if n > 1 else 0.0
    ret_50d = float(close.iloc[-1] / close.iloc[-min(50, n - 1)] - 1) if n > 1 else 0.0

    returns_score = 0.0
    # 20-day return
    if ret_20d > 0.05:
        returns_score += 0.8
    elif ret_20d > 0.02:
        returns_score += 0.4
    elif ret_20d < -0.05:
        returns_score -= 0.8
    elif ret_20d < -0.02:
        returns_score -= 0.4

    # 50-day return adds context
    if ret_50d > 0.08:
        returns_score += 0.3
    elif ret_50d < -0.08:
        returns_score -= 0.3

    returns_score = max(-1.0, min(1.0, returns_score))

    # ── Composite Score ───────────────────────────────────────────────
    # Weighted average of components
    weights = {
        "trend": 0.35,
        "momentum": 0.25,
        "volatility": 0.15,
        "returns": 0.25,
    }
    composite = (
        weights["trend"] * trend_score
        + weights["momentum"] * momentum_score
        + weights["volatility"] * volatility_score
        + weights["returns"] * returns_score
    )

    # Classify
    if composite > 0.15:
        regime = "BULLISH"
    elif composite < -0.15:
        regime = "BEARISH"
    else:
        regime = "SIDEWAYS"

    # Confidence: distance from the neutral zone, scaled to 0-1
    confidence = min(1.0, abs(composite) / 0.8)
    confidence = max(0.10, confidence)  # Floor at 10%

    # Labels
    if abs(trend_score) > 0.7:
        trend_label = "Strong"
    elif abs(trend_score) > 0.3:
        trend_label = "Moderate"
    elif abs(trend_score) > 0.1:
        trend_label = "Weak"
    else:
        trend_label = "Neutral"

    if vol_20 > 0.35:
        vol_label = "Extreme"
    elif vol_20 > 0.25:
        vol_label = "High"
    elif vol_20 > 0.15:
        vol_label = "Moderate"
    else:
        vol_label = "Low"

    # ── Explanation ───────────────────────────────────────────────────
    reasons = []
    if regime == "BULLISH":
        if trend_score > 0.3:
            reasons.append("price is above primary moving averages")
        if momentum_score > 0.2:
            reasons.append("momentum indicators remain positive")
        if returns_score > 0.2:
            reasons.append(f"recent returns are positive ({ret_20d:+.1%} over 20 days)")
        if volatility_score > 0:
            reasons.append("volatility is contained")
    elif regime == "BEARISH":
        if trend_score < -0.3:
            reasons.append("price is below primary moving averages")
        if momentum_score < -0.2:
            reasons.append("momentum indicators are negative")
        if returns_score < -0.2:
            reasons.append(f"recent returns are negative ({ret_20d:+.1%} over 20 days)")
        if volatility_score < -0.2:
            reasons.append("volatility is elevated")
    else:
        reasons.append("trend and momentum signals are mixed")
        if abs(trend_score) < 0.2:
            reasons.append("price is oscillating around its moving averages")
        if abs(returns_score) < 0.2:
            reasons.append("recent returns show no clear direction")

    if not reasons:
        reasons.append("based on a balanced analysis of trend, momentum, and volatility")

    explanation = f"Market classified as {regime} because " + ", ".join(reasons) + "."

    data_warning = ""
    if not has_full_history:
        data_warning = (
            f"Only {n} bars available (recommended: {min_bars}+). "
            "Long-term trend signals (200-day SMA) may be less reliable."
        )

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        trend_label=trend_label,
        volatility_label=vol_label,
        explanation=explanation,
        trend_score=trend_score,
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        breadth_score=0.0,  # Set by breadth analysis
        returns_score=returns_score,
        current_price=current_price,
        sma_50=sma_50_val,
        sma_200=sma_200_val,
        rsi_14=rsi_val,
        realized_vol_20=vol_20,
        recent_return_20d=ret_20d,
        breadth_pct=0.0,
        data_sufficient=True,
        data_warning=data_warning,
    )


def detect_market_regime_with_breadth(
    data_cache: dict[str, pd.DataFrame],
    reference_symbol: str | None = None,
) -> RegimeResult:
    """Detect market regime using a representative symbol plus market breadth.

    Uses the first available symbol (or reference_symbol) for the primary regime
    detection, then enhances with breadth across all cached symbols.

    Parameters
    ----------
    data_cache : dict[str, pd.DataFrame]
        Map of symbol -> OHLCV DataFrame from session state.
    reference_symbol : str, optional
        Primary symbol for regime detection. Defaults to first available.

    Returns
    -------
    RegimeResult
    """
    if not data_cache:
        return RegimeResult(
            regime="UNKNOWN",
            confidence=0.0,
            trend_label="No Data",
            volatility_label="No Data",
            explanation="No market data available for regime detection.",
            data_sufficient=False,
            data_warning="Load market data first.",
        )

    # Pick reference symbol
    if reference_symbol and reference_symbol in data_cache:
        ref_sym = reference_symbol
    else:
        ref_sym = next(iter(data_cache))

    result = detect_regime(data_cache[ref_sym])

    # Market breadth: % of stocks above their 50-day SMA
    above_50sma = 0
    total_counted = 0
    for sym, df in data_cache.items():
        if df is None or df.empty or len(df) < 50:
            continue
        close_val = float(df["close"].iloc[-1])
        sma_50_val = float(ind.sma(df["close"], 50).iloc[-1])
        if pd.notna(sma_50_val) and sma_50_val > 0:
            total_counted += 1
            if close_val > sma_50_val:
                above_50sma += 1

    if total_counted > 0:
        breadth_pct = above_50sma / total_counted
        result.breadth_pct = breadth_pct

        # Breadth score
        breadth_score = (breadth_pct - 0.5) * 2.0  # Maps 0-1 to -1 to +1
        result.breadth_score = breadth_score

        # Refine regime with breadth (10% weight adjustment)
        composite_with_breadth = (
            0.30 * result.trend_score
            + 0.22 * result.momentum_score
            + 0.13 * result.volatility_score
            + 0.22 * result.returns_score
            + 0.13 * breadth_score
        )

        # Re-classify if breadth changes the picture
        if composite_with_breadth > 0.15:
            new_regime = "BULLISH"
        elif composite_with_breadth < -0.15:
            new_regime = "BEARISH"
        else:
            new_regime = "SIDEWAYS"

        if new_regime != result.regime:
            result.regime = new_regime

        # Update confidence
        result.confidence = min(1.0, abs(composite_with_breadth) / 0.8)
        result.confidence = max(0.10, result.confidence)

        # Append breadth info to explanation
        breadth_note = (
            f" Market breadth shows {above_50sma}/{total_counted} "
            f"({breadth_pct:.0%}) stocks trading above their 50-day SMA."
        )
        result.explanation += breadth_note

    return result
