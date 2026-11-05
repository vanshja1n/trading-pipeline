"""Smoke tests for Dashboard components and Streamlit application."""

import pandas as pd
import pytest
from app.components.charts import (
    create_candlestick_chart,
    create_equity_chart,
    create_drawdown_chart,
    create_allocation_donut,
)


def test_chart_components_render():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000.0] * 10,
        },
        index=dates,
    )

    # 1. Candlestick
    fig_candle = create_candlestick_chart(df, symbol="RELIANCE")
    assert fig_candle is not None

    # 2. Equity
    equity = pd.Series([100000.0, 102000.0, 105000.0], index=dates[:3])
    fig_eq = create_equity_chart(equity)
    assert fig_eq is not None

    # 3. Drawdown
    fig_dd = create_drawdown_chart(equity)
    assert fig_dd is not None

    # 4. Allocation
    fig_donut = create_allocation_donut({"RELIANCE": 0.4, "TCS": 0.3, "CASH": 0.3})
    assert fig_donut is not None


def test_all_dashboard_pages_render_without_error(monkeypatch):
    """Verify that every module page in the Streamlit app loads without exceptions."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from streamlit.testing.v1 import AppTest
    from autotrade.data.live import LiveQuote
    from datetime import datetime
    from autotrade.market.constants import IST

    # Prevent external network calls to Yahoo Finance during smoke test
    mock_snap = [
        {
            "Symbol": sym,
            "LTP (₹)": "₹2,500.00",
            "Change": "+1.00%",
            "Day High (₹)": "₹2,550.00",
            "Day Low (₹)": "₹2,450.00",
            "Volume": "100,000",
        }
        for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]
    ]
    monkeypatch.setattr("autotrade.data.ingestion.fetch_market_snapshot", lambda *args, **kwargs: mock_snap)

    mock_quote = LiveQuote(
        symbol="RELIANCE",
        price=2500.0,
        previous_close=2475.0,
        change=25.0,
        change_pct=1.01,
        volume=500000.0,
        status="CLOSED",
        timestamp=datetime.now(IST),
        is_valid=True,
    )
    monkeypatch.setattr(
        "autotrade.data.live.LiveMarketFeed.fetch_single_quote",
        lambda self, sym: LiveQuote(
            symbol=sym,
            price=2000.0,
            previous_close=1980.0,
            change=20.0,
            change_pct=1.0,
            volume=100000.0,
            status="CLOSED",
            timestamp=datetime.now(IST),
            is_valid=True,
        ),
    )

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception

    radio = at.sidebar.radio[0]
    for opt in radio.options:
        if "───" in opt:
            continue
        radio.set_value(opt).run()
        assert not at.exception, f"Error rendering module '{opt}': {at.exception}"
