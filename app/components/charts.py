"""Interactive Plotly financial charts for the Indian Trading Platform."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str = "NSE",
    sma_fast: int = 20,
    sma_slow: int = 50,
) -> go.Figure:
    """Render interactive candlestick chart with volume and SMA overlays."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"{symbol} Price",
            increasing_line_color="#00C805",
            decreasing_line_color="#FF3B30",
        ),
        row=1,
        col=1,
    )

    # 2. Moving Averages
    if len(df) >= sma_fast:
        fast_sma = df["close"].rolling(sma_fast).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=fast_sma,
                line=dict(color="#FF9500", width=1.5),
                name=f"SMA {sma_fast}",
            ),
            row=1,
            col=1,
        )

    if len(df) >= sma_slow:
        slow_sma = df["close"].rolling(sma_slow).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=slow_sma,
                line=dict(color="#0A84FF", width=1.5),
                name=f"SMA {sma_slow}",
            ),
            row=1,
            col=1,
        )

    # 3. Volume
    colors = [
        "#00C805" if c >= o else "#FF3B30"
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} — Price & Volume (NSE)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )
    return fig


def create_equity_chart(
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    title: str = "Portfolio Equity Growth",
    currency_symbol: str = "₹",
) -> go.Figure:
    """Render strategy equity curve against benchmark."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            line=dict(color="#00C805", width=2),
            name="Strategy Equity",
        )
    )

    if benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity.values,
                mode="lines",
                line=dict(color="#8E8E93", width=1.5, dash="dash"),
                name="Benchmark",
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title=f"Equity ({currency_symbol})",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    return fig


def create_drawdown_chart(equity: pd.Series) -> go.Figure:
    """Render underwater drawdown chart."""
    peak = equity.cummax()
    dd = (equity - peak) / peak

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            fill="tozeroy",
            mode="lines",
            line=dict(color="#FF3B30", width=1),
            fillcolor="rgba(255, 59, 48, 0.3)",
            name="Drawdown",
        )
    )

    fig.update_layout(
        title="Underwater Drawdown Profile",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1%",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=30),
        height=300,
    )
    return fig


def create_allocation_donut(allocations: dict[str, float]) -> go.Figure:
    """Render donut chart of portfolio asset allocation."""
    labels = list(allocations.keys())
    values = list(allocations.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo="label+percent",
                marker=dict(colors=["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#64D2FF", "#8E8E93"]),
            )
        ]
    )

    fig.update_layout(
        title="Current Capital Allocation",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig


def create_intraday_chart(
    df: pd.DataFrame,
    symbol: str = "NSE",
    show_vwap: bool = True,
) -> go.Figure:
    """Render interactive intraday candlestick chart with volume and VWAP overlay."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{symbol} — No Intraday Data Available",
            template="plotly_dark",
            height=450,
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"{symbol} 5m Bar",
            increasing_line_color="#00C805",
            decreasing_line_color="#FF3B30",
        ),
        row=1,
        col=1,
    )

    # 2. VWAP overlay
    if show_vwap and "vwap" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["vwap"],
                line=dict(color="#FFD60A", width=1.5, dash="dash"),
                name="Intraday VWAP",
            ),
            row=1,
            col=1,
        )

    # 3. Volume
    colors = [
        "#00C805" if c >= o else "#FF3B30"
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} — Intraday Session Chart (5m Bars + VWAP)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    return fig

