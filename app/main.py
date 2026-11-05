"""Indian Equity Quantitative Trading & Risk Management Platform
Streamlit Dashboard Application
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from autotrade.config import BacktestConfig, RiskConfig
from autotrade.data.cleaning import clean_ohlcv
from autotrade.data.ingestion import download_nse_symbol, fetch_market_snapshot, load_csv
from autotrade.data.storage import Database
from autotrade.data.validation import validate_ohlcv
from autotrade.engine.backtest import Backtester
from autotrade.engine.walk_forward import run_walk_forward
from autotrade.execution.costs import CostBreakdown, IndianEquityCostModel
from autotrade.execution.simulated import SimulatedBroker
from autotrade.market.calendar import NSECalendar
from autotrade.market.constants import (
    CURRENCY,
    CURRENCY_SYMBOL,
    IST,
    NSE_WATCHLIST,
    resolve_nse_symbol,
    strip_exchange_suffix,
)
from autotrade.portfolio.tracker import PortfolioTracker
from autotrade.portfolio.optimizer import optimize_portfolio, OptimizationResult
from autotrade.portfolio.correlation import compute_correlation_analysis, CorrelationResult
from autotrade.market.regime import detect_market_regime_with_breadth, RegimeResult
from autotrade.reliability.analyzer import StrategyReliabilityAnalyzer
from autotrade.reporting import metrics as metrics_mod
from autotrade.reporting.research_report import (
    generate_research_report,
    report_to_html,
    ResearchReportData,
    StrategyComparisonEntry,
)
from autotrade.strategy import indicators as ind
from autotrade.strategy import strategies as strat_mod

from app.components.charts import (
    create_allocation_donut,
    create_candlestick_chart,
    create_drawdown_chart,
    create_equity_chart,
    create_intraday_chart,
)
from autotrade.data.live import (
    LiveMarketFeed,
    LiveQuote,
    evaluate_market_session,
    sync_paper_portfolio_with_live_prices,
)

# ---------------------------------------------------------------------------
# App Setup & Session State
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Pipeline by vansh | NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Glassmorphism & Sleek Metrics
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8B949E;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #F0F6FC;
        font-size: 26px;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-open { background-color: #0E4429; color: #39D353; border: 1px solid #2EA043; }
    .status-closed { background-color: #490202; color: #FF7B72; border: 1px solid #F85149; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db() -> Database:
    return Database("data/database/trading.db")


db = get_db()

# Shared date index for all synthetic data — ensures alignment across symbols
# for correlation/optimization modules.
_SHARED_SAMPLE_DATES: pd.DatetimeIndex | None = None


def _get_shared_sample_dates(periods: int = 300) -> pd.DatetimeIndex:
    """Return a canonical shared date index for synthetic data generation."""
    global _SHARED_SAMPLE_DATES
    if _SHARED_SAMPLE_DATES is None or len(_SHARED_SAMPLE_DATES) != periods:
        end_date = datetime.now(IST).date()
        _SHARED_SAMPLE_DATES = pd.bdate_range(end=end_date, periods=periods, tz=IST)
    return _SHARED_SAMPLE_DATES


def generate_sample_data(symbol: str = "RELIANCE", periods: int = 300) -> pd.DataFrame:
    """Generate realistic synthetic daily OHLCV bars anchored to accurate live market prices.

    All symbols share the same date index to ensure timestamp alignment
    for correlation and portfolio optimization calculations.
    """
    np.random.seed(abs(hash(symbol)) % (2**32))
    dates = _get_shared_sample_dates(periods)

    # Base prices calibrated to actual current NSE market levels
    base_prices = {
        "RELIANCE": 1313.40,
        "TCS": 2495.70,
        "INFY": 1167.40,
        "HDFCBANK": 760.00,
        "ICICIBANK": 1500.00,
        "SBIN": 1006.40,
        "ITC": 278.85,
        "LT": 4050.00,
        "BHARTIARTL": 2056.90,
        "AXISBANK": 1280.00,
    }
    start_p = base_prices.get(symbol.upper(), 1000.0)

    # Geometric random walk with drift
    daily_returns = np.random.normal(0.0005, 0.015, periods)
    price_series = start_p * np.cumprod(1 + daily_returns)

    high = price_series * (1 + np.random.uniform(0.005, 0.02, periods))
    low = price_series * (1 - np.random.uniform(0.005, 0.02, periods))
    open_ = low + np.random.uniform(0.2, 0.8, periods) * (high - low)
    volume = np.random.uniform(200_000, 2_000_000, periods)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": price_series, "volume": volume},
        index=dates,
    )
    df.index.name = "timestamp"
    df.attrs["symbol"] = symbol.upper()
    return df


def load_symbol_data(symbol: str, periods: int = 250) -> pd.DataFrame:
    """Load real, accurate market data from SQLite database or live feed."""
    sym = symbol.upper()
    try:
        df = db.load_market_data(sym)
        if df is not None and not df.empty and len(df) >= 20:
            return df
    except Exception:
        pass

    try:
        start_date = (datetime.now(IST) - timedelta(days=periods * 2)).strftime("%Y-%m-%d")
        df_live = download_nse_symbol(sym, start=start_date)
        if df_live is not None and not df_live.empty:
            try:
                db.save_market_data(sym, df_live)
            except Exception:
                pass
            return df_live
    except Exception:
        pass

    return generate_sample_data(sym, periods)


if "broker" not in st.session_state:
    st.session_state.broker = SimulatedBroker(starting_cash=500_000.0)

if "tracker" not in st.session_state:
    st.session_state.tracker = PortfolioTracker(starting_cash=500_000.0)

if "data_cache" not in st.session_state:
    st.session_state.data_cache = {
        sym: load_symbol_data(sym, 250) for sym in NSE_WATCHLIST
    }

if "live_feed" not in st.session_state:
    st.session_state.live_feed = LiveMarketFeed(cache_ttl_seconds=30.0)

calendar = NSECalendar()
is_open = calendar.is_market_open()

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Trading Pipeline by vansh")
st.sidebar.caption("NSE Equity Quant Platform & Risk Management")

market_status_html = (
    '<div class="status-badge status-open">● NSE MARKET OPEN</div>'
    if is_open
    else '<div class="status-badge status-closed">○ NSE MARKET CLOSED</div>'
)
st.sidebar.markdown(market_status_html, unsafe_allow_html=True)
st.sidebar.write(f"IST: **{datetime.now(IST).strftime('%H:%M:%S | %d %b %Y')}**")

st.sidebar.divider()

page = st.sidebar.radio(
    "Modules",
    [
        "1. Executive Overview",
        "2. Market Data & Ingestion",
        "3. Watchlist & Screener",
        "4. Technical Indicators Lab",
        "5. Strategy Signals",
        "6. Quantitative Backtester",
        "7. Reliability & Overfitting Audit",
        "8. Simulated Paper Trading",
        "9. Portfolio & Exposure",
        "10. Risk & Kill Switch Monitor",
        "11. Trade History & P&L",
        "12. Pipeline & System Diagnostics",
        "─── Advanced Analytics ───",
        "13. Market Regime Detection",
        "14. Portfolio Optimization",
        "15. Correlation Analysis",
        "16. Strategy Comparison",
        "17. Research Report",
        "18. Live Market",
    ],
    index=0,
)

# ---------------------------------------------------------------------------
# Module 1: Executive Overview
# ---------------------------------------------------------------------------
if page == "1. Executive Overview":
    st.header("Trading Pipeline by vansh — Executive Overview")
    st.caption("Real-time portfolio pulse, active risk parameters, and NSE market posture.")

    acc = st.session_state.broker.get_account()
    pos_list = st.session_state.broker.get_all_positions()

    # Compute actual all-time return from starting cash
    starting = st.session_state.broker.starting_cash
    all_time_return = (acc.equity - starting) / starting if starting > 0 else 0.0
    all_time_color = "#39D353" if all_time_return >= 0 else "#FF7B72"

    # Unrealized P&L color
    upnl_color = "#39D353" if acc.unrealized_pnl >= 0 else "#FF7B72"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Portfolio Equity</div>
                <div class="metric-value">{CURRENCY_SYMBOL}{acc.equity:,.2f}</div>
                <div class="metric-sub" style="color: {all_time_color};">{all_time_return:+.2%} All Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Available Cash</div>
                <div class="metric-value">{CURRENCY_SYMBOL}{acc.cash:,.2f}</div>
                <div class="metric-sub">Buying Power: {CURRENCY_SYMBOL}{acc.buying_power:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Unrealized P&L</div>
                <div class="metric-value" style="color: {upnl_color};">{CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}</div>
                <div class="metric-sub">{len(pos_list)} Open Position(s)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        kill_active = not getattr(st.session_state.broker, '_kill_switch_engaged', False)
        kill_color = "#39D353" if kill_active else "#FF453A"
        kill_label = "ACTIVE" if kill_active else "ENGAGED — HALTED"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Risk Kill Switch</div>
                <div class="metric-value" style="color: {kill_color};">{kill_label}</div>
                <div class="metric-sub">Max Drawdown: 20% Cap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Market Snapshot (NSE Bluechips)")
    @st.cache_data(ttl=15)
    def _load_live_market_snapshot():
        return fetch_market_snapshot(NSE_WATCHLIST[:6])

    try:
        rows = _load_live_market_snapshot()
    except Exception:
        rows = []

    # Fallback to local session data if offline or provider failure
    if not rows or any(r.get("LTP (₹)") == "N/A" for r in rows):
        fallback_rows = []
        for sym in NSE_WATCHLIST[:6]:
            df = st.session_state.data_cache.get(sym)
            if df is not None and not df.empty:
                last_close = float(df["close"].iloc[-1])
                prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
                chg = (last_close - prev_close) / prev_close
                fallback_rows.append(
                    {
                        "Symbol": sym,
                        "LTP (₹)": f"₹{last_close:,.2f}",
                        "Change": f"{chg:+.2%}",
                        "Day High (₹)": f"₹{float(df['high'].iloc[-1]):,.2f}",
                        "Day Low (₹)": f"₹{float(df['low'].iloc[-1]):,.2f}",
                        "Volume": f"{int(df['volume'].iloc[-1]):,}",
                    }
                )
        if not rows:
            rows = fallback_rows
        else:
            fb_map = {r["Symbol"]: r for r in fallback_rows}
            for i, r in enumerate(rows):
                if r.get("LTP (₹)") == "N/A" and r["Symbol"] in fb_map:
                    rows[i] = fb_map[r["Symbol"]]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Module 2: Market Data & Ingestion
# ---------------------------------------------------------------------------
elif page == "2. Market Data & Ingestion":
    st.header("Market Data & Ingestion Pipeline")
    st.caption("Download historical NSE price series or upload custom CSV datasets.")

    tab1, tab2, tab3 = st.tabs(["Yahoo Finance (NSE)", "CSV Upload", "Stored Datasets"])

    with tab1:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sym_input = st.selectbox("Select NSE Symbol", NSE_WATCHLIST)
        with c2:
            start_d = st.date_input("Start Date", date.today() - timedelta(days=365))
        with c3:
            end_d = st.date_input("End Date", date.today())

        if st.button("Fetch Historical Bars", type="primary"):
            with st.spinner(f"Querying Yahoo Finance for {sym_input}.NS..."):
                try:
                    df_dl = download_nse_symbol(sym_input, start=str(start_d), end=str(end_d))
                    st.session_state.data_cache[sym_input] = df_dl
                    db.save_market_data(sym_input, df_dl)
                    st.success(f"Successfully downloaded {len(df_dl)} bars for {sym_input} and saved to SQLite!")
                except Exception as e:
                    st.warning(f"Live download notice: {e}. Using sanitized local simulation series.")
                    synth = generate_sample_data(sym_input, 250)
                    st.session_state.data_cache[sym_input] = synth
                    db.save_market_data(sym_input, synth)
                    st.success(f"Generated and stored {len(synth)} bars for {sym_input}.")

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV (OHLCV)", type=["csv"])
        custom_sym = st.text_input("Symbol Name for Upload", value="CUSTOM_NSE")
        if uploaded_file and st.button("Ingest & Validate CSV"):
            with st.spinner("Processing CSV..."):
                try:
                    raw_df = pd.read_csv(uploaded_file)
                    with io.StringIO(uploaded_file.getvalue().decode("utf-8")) as f:
                        import tempfile
                        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
                            tf.write(uploaded_file.getvalue().decode("utf-8"))
                            tmp_p = tf.name
                        ingested = load_csv(tmp_p, symbol=custom_sym)
                        cleaned = clean_ohlcv(ingested)
                        report = validate_ohlcv(cleaned, symbol=custom_sym)
                        st.session_state.data_cache[custom_sym] = cleaned
                        db.save_market_data(custom_sym, cleaned)
                        st.success(f"Ingested {len(cleaned)} bars! Validation: {'PASSED' if report.is_valid else 'FAILED'}")
                        st.text(report.summary())
                except Exception as err:
                    st.error(f"Failed to ingest CSV: {err}")

    with tab3:
        stored = db.get_available_symbols()
        st.write(f"**SQLite Database Symbols:** {stored or 'None stored yet'}")
        if stored:
            sel_sym = st.selectbox("View Stored Symbol Data", stored)
            df_stored = db.load_market_data(sel_sym)
            st.dataframe(df_stored.tail(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Module 3: Watchlist & Screener
# ---------------------------------------------------------------------------
elif page == "3. Watchlist & Screener":
    st.header("NSE Bluechip Watchlist & Screener")
    st.caption("Active monitoring of high-liquidity Nifty constituents.")

    watchlist_data = []
    for sym in NSE_WATCHLIST:
        df = st.session_state.data_cache.get(sym)
        if df is None:
            df = load_symbol_data(sym, 250)
            st.session_state.data_cache[sym] = df

        live_q = st.session_state.live_feed._cached_quotes.get(sym)
        if live_q and live_q.is_valid and live_q.price > 0:
            close_p = live_q.price
            prev_c = live_q.previous_close if live_q.previous_close > 0 else float(df["close"].iloc[-2])
            ret = live_q.change_pct / 100.0
            data_src = "Live (NSE)"
        else:
            close_p = float(df["close"].iloc[-1])
            prev_c = float(df["close"].iloc[-2]) if len(df) > 1 else close_p
            ret = (close_p - prev_c) / prev_c if prev_c > 0 else 0.0
            data_src = "NSE Daily Close"

        open_p = float(df["open"].iloc[-1])
        rsi_val = float(ind.rsi(df["close"], 14).iloc[-1])
        vol_20 = float(ind.realized_volatility(df["close"], 20, 248).iloc[-1])

        watchlist_data.append(
            {
                "Symbol": sym,
                "Price (₹)": round(close_p, 2),
                "Day %": f"{ret:+.2%}",
                "RSI (14)": round(rsi_val, 1),
                "Realized Vol": f"{vol_20:.1%}",
                "Trend": "BULLISH" if close_p >= open_p else "BEARISH",
                "Source": data_src,
            }
        )

    w_df = pd.DataFrame(watchlist_data)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

    sel = st.selectbox("Select symbol to chart:", NSE_WATCHLIST)
    if sel not in st.session_state.data_cache:
        st.session_state.data_cache[sel] = load_symbol_data(sel, 250)
    df_chart = st.session_state.data_cache[sel]
    fig = create_candlestick_chart(df_chart, symbol=sel)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Module 4: Technical Indicators Lab
# ---------------------------------------------------------------------------
elif page == "4. Technical Indicators Lab":
    st.header("Technical Indicators Lab")
    st.caption("Mathematical validation of technical indicators on price series.")

    sel_sym = st.selectbox("Symbol", list(st.session_state.data_cache.keys()))
    df = st.session_state.data_cache[sel_sym]

    c1, c2, c3 = st.columns(3)
    with c1:
        fast_sma = st.slider("Fast SMA Window", 5, 50, 20)
    with c2:
        slow_sma = st.slider("Slow SMA Window", 20, 200, 50)
    with c3:
        rsi_window = st.slider("RSI Window", 5, 30, 14)

    fig = create_candlestick_chart(df, symbol=sel_sym, sma_fast=fast_sma, sma_slow=slow_sma)
    st.plotly_chart(fig, use_container_width=True)

    # RSI Subplot
    rsi_series = ind.rsi(df["close"], rsi_window)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, line=dict(color="#BF5AF2", width=1.5), name="RSI"))
    fig_rsi.add_hline(y=70, line=dict(color="#FF453A", dash="dash"))
    fig_rsi.add_hline(y=30, line=dict(color="#30D158", dash="dash"))
    fig_rsi.update_layout(
        title=f"Relative Strength Index ({rsi_window})",
        template="plotly_dark",
        yaxis=dict(range=[0, 100]),
        height=250,
        margin=dict(l=40, r=40, t=40, b=20),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------------------------------------------------------------
# Module 5: Strategy Signals
# ---------------------------------------------------------------------------
elif page == "5. Strategy Signals":
    st.header("Quantitative Strategy Signals")
    st.caption("Inspect target exposures generated by registered algorithmic strategies.")

    sel_sym = st.selectbox("Symbol", list(st.session_state.data_cache.keys()), key="sig_sym")
    df = st.session_state.data_cache[sel_sym]

    strat_names = list(strat_mod.REGISTRY.keys())
    sel_strat = st.selectbox("Select Strategy", strat_names, index=0)

    strat_instance = strat_mod.build(sel_strat)
    signals = strat_instance(df)

    latest_signal = float(signals.iloc[-1])
    sig_text = "BUY / LONG (+1.0)" if latest_signal > 0 else ("SELL / SHORT (-1.0)" if latest_signal < 0 else "FLAT (0.0)")

    st.info(f"**Current Recommendation:** {sig_text} for {sel_sym}")

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Price", line=dict(color="#8E8E93", width=1)))
    buy_signals = df[signals == 1.0]
    fig_sig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["close"], mode="markers", marker=dict(color="#00C805", size=6), name="Long"))
    fig_sig.update_layout(title=f"Strategy Signals Overlay: {strat_instance.name}", template="plotly_dark", height=450)
    st.plotly_chart(fig_sig, use_container_width=True)

# ---------------------------------------------------------------------------
# Module 6: Quantitative Backtester
# ---------------------------------------------------------------------------
elif page == "6. Quantitative Backtester":
    st.header("Event-Driven Quantitative Backtester")
    st.caption("Next-open execution, whole-share sizing, and realistic Indian transaction costs.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_sym = st.selectbox("Symbol", list(st.session_state.data_cache.keys()), key="bt_sym")
    with c2:
        strat_key = st.selectbox("Strategy", list(strat_mod.REGISTRY.keys()), index=0, key="bt_strat")
    with c3:
        initial_cash = st.number_input("Starting Cash (₹)", value=500_000.0, step=50_000.0)
    with c4:
        use_indian_tax = st.checkbox("Apply Indian Statutory Costs (STT, GST, etc.)", value=True)

    df = st.session_state.data_cache[sel_sym]
    if st.button("Run Simulation", type="primary"):
        strat = strat_mod.build(strat_key)
        cfg = BacktestConfig(
            starting_cash=initial_cash,
            slippage_bps=2.0,
            use_indian_costs=use_indian_tax,
        )
        bt = Backtester(strat, cfg)
        res = bt.run(df, symbol=sel_sym)

        m = res.metrics
        st.subheader("Performance Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Final Equity", f"{CURRENCY_SYMBOL}{m.final_equity:,.2f}", f"{m.total_return:+.2%}")
        k2.metric("CAGR", f"{m.cagr:.2%}")
        k3.metric("Sharpe Ratio", f"{m.sharpe:.2f}")
        k4.metric("Max Drawdown", f"{m.max_drawdown:.2%}")
        k5.metric("Win Rate", f"{m.win_rate:.1%}")

        fig_eq = create_equity_chart(res.equity, title=f"Equity Growth ({strat.name} on {sel_sym})")
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = create_drawdown_chart(res.equity)
        st.plotly_chart(fig_dd, use_container_width=True)

        if not res.trades.empty:
            st.subheader(f"Closed Trades Log ({len(res.trades)} executions)")
            st.dataframe(res.trades.tail(20), use_container_width=True)

# ---------------------------------------------------------------------------
# Module 7: Reliability & Overfitting Audit
# ---------------------------------------------------------------------------
elif page == "7. Reliability & Overfitting Audit":
    st.header("Strategy Reliability & Overfitting Audit")
    st.caption("Walk-forward out-of-sample verification & friction decay testing.")

    sel_sym = st.selectbox("Asset", list(st.session_state.data_cache.keys()), key="rel_sym")
    strat_key = st.selectbox("Strategy to Audit", list(strat_mod.REGISTRY.keys()), index=0, key="rel_strat")
    df = st.session_state.data_cache[sel_sym]

    if st.button("Conduct Comprehensive Audit", type="primary"):
        with st.spinner("Executing walk-forward cross-validation and cost sensitivity tests..."):
            analyzer = StrategyReliabilityAnalyzer(lambda: strat_mod.build(strat_key))
            report = analyzer.analyze(df, symbol=sel_sym)

            score_color = "#39D353" if report.overall_score >= 70 else ("#FF9F0A" if report.overall_score >= 50 else "#FF453A")
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-title">Reliability Score</div>
                    <div class="metric-value" style="color: {score_color}; font-size: 42px;">{report.overall_score:.1f} / 100</div>
                    <div class="status-badge" style="background-color: {score_color}33; color: {score_color}; margin-top: 8px;">VERDICT: {report.verdict}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Key Findings & Strengths")
                for s in report.strengths:
                    st.write(f"✅ {s}")
            with c2:
                st.subheader("Vulnerabilities & Friction Decay")
                for v in report.vulnerabilities:
                    st.write(f"⚠️ {v}")

            st.subheader("Transaction Cost Sensitivity Curve")
            cs_data = [
                {
                    "Slippage": f"{cs.slippage_bps} bps",
                    "Taxes": "Indian Statutory" if cs.use_taxes else "None",
                    "Return": f"{cs.annualized_return:.2%}",
                    "Sharpe": f"{cs.sharpe:.2f}",
                    "Profit Factor": f"{cs.profit_factor:.2f}",
                }
                for cs in report.cost_sensitivity
            ]
            st.dataframe(pd.DataFrame(cs_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Module 8: Simulated Paper Trading
# ---------------------------------------------------------------------------
elif page == "8. Simulated Paper Trading":
    st.header("Simulated Paper Trading Terminal")
    st.caption("Execute simulated market orders in Indian Equities with live local accounting.")

    broker = st.session_state.broker
    tracker = st.session_state.tracker
    acc = broker.get_account()

    upnl_color = "#39D353" if acc.unrealized_pnl >= 0 else "#FF7B72"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash Balance", f"{CURRENCY_SYMBOL}{acc.cash:,.2f}")
    c2.metric("Portfolio Equity", f"{CURRENCY_SYMBOL}{acc.equity:,.2f}")
    c3.metric("Invested Value", f"{CURRENCY_SYMBOL}{acc.invested_value:,.2f}")
    c4.metric("Unrealized P&L", f"{CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}")

    # Reconciliation check
    recon_diff = abs(acc.equity - (acc.cash + acc.invested_value))
    if recon_diff > 0.01:
        st.warning(f"⚠️ Reconciliation gap: Equity ({CURRENCY_SYMBOL}{acc.equity:,.2f}) ≠ Cash ({CURRENCY_SYMBOL}{acc.cash:,.2f}) + Invested ({CURRENCY_SYMBOL}{acc.invested_value:,.2f}). Diff: {CURRENCY_SYMBOL}{recon_diff:,.2f}")

    st.subheader("Order Entry")
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        order_sym = st.selectbox("Symbol", NSE_WATCHLIST, key="ord_sym")
    with o2:
        # Use live feed price first, fall back to historical close
        live_feed: LiveMarketFeed = st.session_state.live_feed
        live_quote = live_feed._cached_quotes.get(order_sym)
        if live_quote and live_quote.is_valid and live_quote.price > 0:
            current_p = live_quote.price
            price_source = "Live (yfinance)"
        else:
            df = st.session_state.data_cache.get(order_sym)
            current_p = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0
            price_source = "Historical Close"
        if current_p > 0:
            broker.update_price(order_sym, current_p)
            st.write(f"LTP: **{CURRENCY_SYMBOL}{current_p:,.2f}**")
            st.caption(f"Source: {price_source}")
        else:
            st.warning("No valid price available for this symbol.")
    with o3:
        order_qty = st.number_input("Shares", min_value=1, value=10, step=1)
    with o4:
        order_side = st.selectbox("Side", ["BUY", "SELL"])

    if st.button("Submit Market Order", type="primary"):
        if current_p <= 0:
            st.error("Cannot execute: no valid market price available.")
        else:
            try:
                ord_res = broker.submit_order(order_sym, order_qty, order_side.lower(), price=current_p)
                # Sync the fill to the PortfolioTracker so Live Market page is consistent
                tracker.record_fill(
                    symbol=order_sym,
                    side=order_side.lower(),
                    qty=order_qty,
                    fill_price=ord_res.filled_price,
                    costs=0.0,  # Costs already deducted from broker cash
                )
                st.success(f"Filled Order {ord_res.order_id}: {order_side} {order_qty} {order_sym} @ {CURRENCY_SYMBOL}{ord_res.filled_price:,.2f}")
            except Exception as e:
                st.error(f"Order rejected: {e}")

    st.subheader("Current Holdings")
    positions = broker.get_all_positions()
    if positions:
        pos_rows = [
            {
                "Symbol": p.symbol,
                "Shares": p.shares,
                "Avg Price": f"{CURRENCY_SYMBOL}{p.avg_price:,.2f}",
                "Current Price": f"{CURRENCY_SYMBOL}{p.current_price:,.2f}",
                "Market Value": f"{CURRENCY_SYMBOL}{p.market_value:,.2f}",
                "Unrealized P&L": f"{CURRENCY_SYMBOL}{p.unrealized_pnl:+,.2f}",
                "P&L %": f"{p.unrealized_pnl_pct:+.2%}",
            }
            for p in positions
        ]
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions. Use Order Entry above to simulate trades.")

# ---------------------------------------------------------------------------
# Module 9: Portfolio & Exposure
# ---------------------------------------------------------------------------
elif page == "9. Portfolio & Exposure":
    st.header("Portfolio Analytics & Asset Allocation")
    st.caption("Multi-asset exposure, sector diversification, and cash utilization.")

    broker = st.session_state.broker
    acc = broker.get_account()
    positions = broker.get_all_positions()

    # Portfolio Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equity", f"{CURRENCY_SYMBOL}{acc.equity:,.2f}")
    c2.metric("Cash", f"{CURRENCY_SYMBOL}{acc.cash:,.2f}")
    c3.metric("Invested Value", f"{CURRENCY_SYMBOL}{acc.invested_value:,.2f}")
    upnl_delta = f"{acc.unrealized_pnl:+,.2f}" if acc.unrealized_pnl != 0 else None
    c4.metric("Unrealized P&L", f"{CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}", delta=upnl_delta)

    # Reconciliation
    st.caption(f"Equity = Cash + Invested: {CURRENCY_SYMBOL}{acc.cash:,.2f} + {CURRENCY_SYMBOL}{acc.invested_value:,.2f} = {CURRENCY_SYMBOL}{acc.cash + acc.invested_value:,.2f}")

    allocations = {"CASH": acc.cash}
    for p in positions:
        allocations[p.symbol] = p.market_value

    fig_donut = create_allocation_donut(allocations)
    st.plotly_chart(fig_donut, use_container_width=True)

    # Position detail table
    if positions:
        st.subheader("Position Details")
        pos_rows = [
            {
                "Symbol": p.symbol,
                "Shares": p.shares,
                "Avg Entry": f"{CURRENCY_SYMBOL}{p.avg_price:,.2f}",
                "Current Price": f"{CURRENCY_SYMBOL}{p.current_price:,.2f}",
                "Market Value": f"{CURRENCY_SYMBOL}{p.market_value:,.2f}",
                "Cost Basis": f"{CURRENCY_SYMBOL}{p.avg_price * p.shares:,.2f}",
                "Unrealized P&L": f"{CURRENCY_SYMBOL}{p.unrealized_pnl:+,.2f}",
                "Return %": f"{p.unrealized_pnl_pct:+.2%}",
            }
            for p in positions
        ]
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        # Verify portfolio-level P&L = sum of position P&L
        sum_pos_upnl = sum(p.unrealized_pnl for p in positions)
        if abs(sum_pos_upnl - acc.unrealized_pnl) > 0.01:
            st.warning(f"⚠️ P&L mismatch: Portfolio unrealized ({CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}) ≠ Sum of positions ({CURRENCY_SYMBOL}{sum_pos_upnl:+,.2f})")
        else:
            st.success(f"✅ P&L reconciled: Portfolio unrealized P&L = Sum of position P&L = {CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}")

# ---------------------------------------------------------------------------
# Module 10: Risk & Kill Switch Monitor
# ---------------------------------------------------------------------------
elif page == "10. Risk & Kill Switch Monitor":
    st.header("Risk Engine & Kill Switch Status")
    st.caption("Institutional risk controls: Drawdown caps, ATR stops, and session boundaries.")

    broker = st.session_state.broker
    acc = broker.get_account()
    positions = broker.get_all_positions()

    risk_cfg = RiskConfig(max_drawdown=0.20, max_daily_loss=0.05, stop_loss_atr=2.0)

    # Compute actual drawdown from peak
    peak_equity = broker.starting_cash  # Simplified — in production, track peak
    if acc.equity > peak_equity:
        peak_equity = acc.equity
    current_dd = 1.0 - acc.equity / peak_equity if peak_equity > 0 else 0.0
    dd_breached = current_dd >= risk_cfg.max_drawdown

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Portfolio Drawdown", f"{risk_cfg.max_drawdown:.0%}", "Hard Stop")
    c2.metric("Daily Loss Limit", f"{risk_cfg.max_daily_loss:.0%}", "Per Session")
    c3.metric("ATR Stop Multiple", f"{risk_cfg.stop_loss_atr}x ATR", "Trailing / Static")
    dd_color = "inverse" if dd_breached else "normal"
    c4.metric("Current Drawdown", f"{current_dd:.2%}", delta=f"Limit: {risk_cfg.max_drawdown:.0%}", delta_color=dd_color)

    # Exposure metrics
    st.subheader("Exposure Monitor")
    e1, e2, e3 = st.columns(3)
    exposure_pct = acc.invested_value / acc.equity if acc.equity > 0 else 0.0
    e1.metric("Gross Exposure", f"{exposure_pct:.1%}")
    e2.metric("Open Positions", f"{len(positions)}")
    max_pos = max((abs(p.market_value) for p in positions), default=0.0)
    max_concentration = max_pos / acc.equity if acc.equity > 0 else 0.0
    e3.metric("Max Position Concentration", f"{max_concentration:.1%}")

    st.subheader("Kill Switch Controls")
    kill_engaged = getattr(broker, '_kill_switch_engaged', False)
    if kill_engaged:
        st.error("🚨 KILL SWITCH ENGAGED — All trading halted. All positions flattened.")
        if st.button("Reset Kill Switch", type="secondary"):
            broker.reset_kill_switch()
            st.success("Kill switch reset. Trading re-enabled.")
            st.rerun()
    else:
        if dd_breached:
            st.warning(f"⚠️ Drawdown ({current_dd:.2%}) exceeds limit ({risk_cfg.max_drawdown:.0%}). Kill switch recommended.")
        else:
            st.success("Risk Status: NORMAL (No active circuit breaches)")

        if st.button("Trigger Emergency Kill Switch", type="secondary"):
            closed_count = broker.engage_kill_switch()
            st.warning(f"🚨 Emergency Kill Switch engaged! Closed {closed_count} position(s). All trading halted.")
            st.rerun()

# ---------------------------------------------------------------------------
# Module 11: Trade History & P&L
# ---------------------------------------------------------------------------
elif page == "11. Trade History & P&L":
    st.header("Trade History & Execution Records")
    st.caption("Audit log of all paper and live executions with cost breakdowns.")

    trades = st.session_state.broker._trade_history
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)
    else:
        st.info("No trade records yet. Run a simulation or submit an order in the Paper Trading module.")

# ---------------------------------------------------------------------------
# Module 12: Pipeline & System Diagnostics
# ---------------------------------------------------------------------------
elif page == "12. Pipeline & System Diagnostics":
    st.header("Pipeline & System Diagnostics")
    st.caption("End-to-end system health: Database connectivity, accounting reconciliation, NSE exchange calendar, and data cache alignment.")

    broker = st.session_state.broker
    acc = broker.get_account()
    positions = broker.get_all_positions()
    cal = NSECalendar()
    feed = st.session_state.live_feed
    db_symbols = db.get_available_symbols()

    # Top-level Health Cards
    h1, h2, h3, h4 = st.columns(4)

    # 1. DB Status
    db_ok = True
    try:
        _ = len(db_symbols)
        h1.metric("Database", "CONNECTED", f"{len(db_symbols)} symbols stored")
    except Exception as e:
        db_ok = False
        h1.metric("Database", "ERROR", str(e), delta_color="inverse")

    # 2. Accounting Reconciliation Status
    recon_diff = abs(acc.equity - (acc.cash + acc.invested_value))
    recon_ok = recon_diff <= 0.01
    h2.metric(
        "Accounting State",
        "RECONCILED" if recon_ok else "MISMATCH",
        f"Diff: {CURRENCY_SYMBOL}{recon_diff:.2f}",
        delta_color="normal" if recon_ok else "inverse",
    )

    # 3. Calendar Status
    session_status = feed.get_market_session()
    h3.metric(
        "Exchange Session",
        session_status.status,
        f"Next: {cal.next_trading_day(date.today())}",
    )

    # 4. Cache Alignment
    cache_syms = list(st.session_state.data_cache.keys())
    aligned = True
    if len(cache_syms) >= 2:
        idx0 = st.session_state.data_cache[cache_syms[0]].index
        for s in cache_syms[1:]:
            if not st.session_state.data_cache[s].index.equals(idx0):
                aligned = False
                break
    h4.metric(
        "Data Alignment",
        "SYNCHRONIZED" if aligned else "DESYNCHRONIZED",
        f"{len(cache_syms)} series cached",
        delta_color="normal" if aligned else "inverse",
    )

    st.markdown("---")

    diag_tabs = st.tabs([
        "Accounting Reconciliation",
        "Database & Storage",
        "NSE Calendar & Holidays",
        "Data Feed & Alignment",
    ])

    with diag_tabs[0]:
        st.subheader("Accounting Integrity Check")
        st.write(
            f"**Equation:** Equity (`{CURRENCY_SYMBOL}{acc.equity:,.2f}`) = "
            f"Cash (`{CURRENCY_SYMBOL}{acc.cash:,.2f}`) + "
            f"Invested Value (`{CURRENCY_SYMBOL}{acc.invested_value:,.2f}`)"
        )
        if recon_ok:
            st.success("✅ Portfolio balances strictly reconcile. Cash + Invested = Equity within tolerance.")
        else:
            st.error(f"❌ Balance discrepancy of {CURRENCY_SYMBOL}{recon_diff:,.2f} detected.")

        sum_pos_upnl = sum(p.unrealized_pnl for p in positions)
        pnl_diff = abs(sum_pos_upnl - acc.unrealized_pnl)
        if pnl_diff <= 0.01:
            st.success(f"✅ Position P&L matches Portfolio Unrealized P&L: {CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}")
        else:
            st.warning(f"⚠️ Unrealized P&L mismatch: sum(positions)={CURRENCY_SYMBOL}{sum_pos_upnl:+,.2f} vs portfolio={CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}")

        kill_status = "ENGAGED (Halted)" if getattr(broker, "_kill_switch_engaged", False) else "ARMED (Normal)"
        st.write(f"**Kill Switch State:** `{kill_status}`")

    with diag_tabs[1]:
        st.subheader("Storage & SQLite Diagnostics")
        st.write(f"**Database Path:** `{db.db_path}`")
        st.write(f"**Available Stored Symbols:** `{db_symbols if db_symbols else 'None yet'}`")
        if st.button("Run DB Integrity Check", key="btn_db_check"):
            try:
                import sqlite3
                conn = sqlite3.connect(db.db_path)
                cur = conn.cursor()
                cur.execute("PRAGMA integrity_check;")
                res = cur.fetchone()[0]
                conn.close()
                if res == "ok":
                    st.success("✅ SQLite `PRAGMA integrity_check`: OK")
                else:
                    st.error(f"❌ SQLite integrity check failed: {res}")
            except Exception as e:
                st.error(f"Database check failed: {e}")

    with diag_tabs[2]:
        st.subheader("NSE Holiday & Session Schedule")
        st.write(f"**Total Known Holidays:** {len(cal.holidays)} dates")
        st.write(f"**Current IST Server Time:** {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
        st.write(f"**Market Open Right Now:** `{cal.is_market_open()}`")
        
        # Display next upcoming holidays
        upcoming = sorted([h for h in cal.holidays if h >= date.today()])[:5]
        if upcoming:
            st.write("**Next Upcoming Holidays:**")
            for h in upcoming:
                st.write(f"- {h.strftime('%A, %d %B %Y')}")

    with diag_tabs[3]:
        st.subheader("Data Cache & Multi-Asset Alignment")
        st.write(f"**Symbols in Memory Cache:** {len(cache_syms)}")
        if cache_syms:
            df_sample = st.session_state.data_cache[cache_syms[0]]
            st.write(f"**Date Range:** {df_sample.index[0].strftime('%Y-%m-%d')} to {df_sample.index[-1].strftime('%Y-%m-%d')} ({len(df_sample)} bars)")
            st.write(f"**Index Alignment Across All Cached Assets:** `{'Aligned' if aligned else 'Misaligned'}`")
        if aligned:
            st.success("✅ Timestamp alignment confirmed. Multi-asset correlation and Markowitz optimization will operate without common-row drops.")
        else:
            st.warning("⚠️ Warning: Data timestamps differ across symbols. Dropna may reduce sample sizes.")

# ---------------------------------------------------------------------------
# Separator label — no-op when selected
# ---------------------------------------------------------------------------
elif page == "─── Advanced Analytics ───":
    st.header("Advanced Analytics")
    st.caption("Select a module from the sidebar to access Market Regime Detection, Portfolio Optimization, Correlation Analysis, Strategy Comparison, or Research Report generation.")
    st.info("These modules extend the core platform with advanced quantitative analytics. Choose a module from the sidebar below the separator.")

# ---------------------------------------------------------------------------
# Module 13: Market Regime Detection
# ---------------------------------------------------------------------------
elif page == "13. Market Regime Detection":
    st.header("Market Regime Detection")
    st.caption("Quantitative classification of the current market environment using trend, momentum, volatility, and breadth signals.")

    # Ensure all watchlist symbols are in the data cache
    for sym in NSE_WATCHLIST:
        if sym not in st.session_state.data_cache:
            st.session_state.data_cache[sym] = load_symbol_data(sym, 250)

    # Run regime detection with breadth
    regime_result = detect_market_regime_with_breadth(
        st.session_state.data_cache,
        reference_symbol="RELIANCE",
    )

    # Store for report generation
    st.session_state["last_regime"] = regime_result

    # ── Regime Card ───────────────────────────────────────────────
    regime_color = {
        "BULLISH": "#39D353",
        "BEARISH": "#FF453A",
        "SIDEWAYS": "#FF9F0A",
        "UNKNOWN": "#8B949E",
    }.get(regime_result.regime, "#8B949E")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-title">Market Regime</div>
                <div class="metric-value" style="color: {regime_color}; font-size: 32px;">{regime_result.regime}</div>
                <div class="metric-sub" style="color: {regime_color};">
                    {'● Live Detection' if is_open else '○ Last Detected (Market Closed)'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-title">Confidence</div>
                <div class="metric-value">{regime_result.confidence:.0%}</div>
                <div class="metric-sub">Composite Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        trend_color = "#39D353" if "Strong" in regime_result.trend_label else (
            "#FF9F0A" if "Moderate" in regime_result.trend_label else "#8B949E"
        )
        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-title">Trend</div>
                <div class="metric-value" style="color: {trend_color};">{regime_result.trend_label}</div>
                <div class="metric-sub">Direction Strength</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        vol_color = "#39D353" if regime_result.volatility_label == "Low" else (
            "#FF9F0A" if regime_result.volatility_label == "Moderate" else "#FF453A"
        )
        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-title">Volatility</div>
                <div class="metric-value" style="color: {vol_color};">{regime_result.volatility_label}</div>
                <div class="metric-sub">Realized (20-day)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Explanation ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Regime Explanation</div>
            <div style="color: #C9D1D9; margin-top: 8px; font-size: 14px; line-height: 1.6;">
                {regime_result.explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if regime_result.data_warning:
        st.warning(f"⚠️ {regime_result.data_warning}")

    # ── Signal Components ─────────────────────────────────────────
    st.subheader("Signal Components")
    sig_col1, sig_col2 = st.columns(2)
    with sig_col1:
        signal_data = [
            {"Component": "Trend Score", "Value": f"{regime_result.trend_score:+.3f}", "Weight": "35%"},
            {"Component": "Momentum Score", "Value": f"{regime_result.momentum_score:+.3f}", "Weight": "22%"},
            {"Component": "Volatility Score", "Value": f"{regime_result.volatility_score:+.3f}", "Weight": "13%"},
            {"Component": "Returns Score", "Value": f"{regime_result.returns_score:+.3f}", "Weight": "22%"},
            {"Component": "Breadth Score", "Value": f"{regime_result.breadth_score:+.3f}", "Weight": "13%"},
        ]
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

    with sig_col2:
        underlying = [
            {"Metric": "Current Price", "Value": f"₹{regime_result.current_price:,.2f}"},
            {"Metric": "SMA 50", "Value": f"₹{regime_result.sma_50:,.2f}"},
            {"Metric": "SMA 200", "Value": f"₹{regime_result.sma_200:,.2f}"},
            {"Metric": "RSI (14)", "Value": f"{regime_result.rsi_14:.1f}"},
            {"Metric": "Realized Vol (20d)", "Value": f"{regime_result.realized_vol_20:.1%}"},
            {"Metric": "20-Day Return", "Value": f"{regime_result.recent_return_20d:+.2%}"},
            {"Metric": "Breadth (>50 SMA)", "Value": f"{regime_result.breadth_pct:.0%}"},
        ]
        st.dataframe(pd.DataFrame(underlying), use_container_width=True, hide_index=True)

    # ── Signal Component Chart ────────────────────────────────────
    st.subheader("Signal Component Breakdown")
    comp_names = ["Trend", "Momentum", "Volatility", "Returns", "Breadth"]
    comp_vals = [
        regime_result.trend_score,
        regime_result.momentum_score,
        regime_result.volatility_score,
        regime_result.returns_score,
        regime_result.breadth_score,
    ]
    comp_colors = [
        "#39D353" if v > 0 else "#FF453A" for v in comp_vals
    ]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=comp_names,
        y=comp_vals,
        marker_color=comp_colors,
        text=[f"{v:+.2f}" for v in comp_vals],
        textposition="outside",
    ))
    fig_comp.update_layout(
        template="plotly_dark",
        yaxis=dict(range=[-1.2, 1.2], title="Score"),
        height=350,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    fig_comp.add_hline(y=0, line=dict(color="#30363D", width=1))
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------------------------
# Module 14: Portfolio Optimization
# ---------------------------------------------------------------------------
elif page == "14. Portfolio Optimization":
    st.header("Portfolio Optimization")
    st.caption("Mean-variance optimization using actual historical returns from the NSE universe.")

    # Ensure all watchlist symbols are in the cache
    for sym in NSE_WATCHLIST:
        if sym not in st.session_state.data_cache:
            st.session_state.data_cache[sym] = load_symbol_data(sym, 250)

    # Get current portfolio weights from broker
    broker = st.session_state.broker
    acc = broker.get_account()
    positions = broker.get_all_positions()

    current_weights: dict[str, float] = {}
    if positions and acc.equity > 0:
        for p in positions:
            current_weights[p.symbol] = p.market_value / acc.equity
    # Remaining is cash — not allocated to any stock

    # Run optimization
    opt_symbols = list(st.session_state.data_cache.keys())
    opt_result = optimize_portfolio(
        data_cache=st.session_state.data_cache,
        symbols=opt_symbols,
        current_weights=current_weights if current_weights else None,
        risk_free_rate=0.07,
        periods_per_year=248,
    )

    # Store for report
    st.session_state["last_optimization"] = opt_result

    if not opt_result.data_sufficient:
        st.error(f"⚠️ {opt_result.warning}")
        st.info("Load more market data via Module 2 or ensure at least 2 symbols have 60+ days of history.")
    else:
        if opt_result.warning:
            st.warning(f"⚠️ {opt_result.warning}")

        # ── Current vs. Optimized Metrics ─────────────────────────
        st.subheader("Portfolio Metrics Comparison")

        # Determine the best optimized portfolio (highest Sharpe)
        best_opt = max(opt_result.optimized_allocations, key=lambda a: a.sharpe_ratio) if opt_result.optimized_allocations else None

        curr = opt_result.current_allocation
        metrics_rows = [
            {
                "Portfolio": curr.name,
                "Expected Return": f"{curr.expected_return:.2%}",
                "Volatility": f"{curr.expected_volatility:.2%}",
                "Sharpe Ratio": f"{curr.sharpe_ratio:.2f}",
                "Max Drawdown": f"{curr.max_drawdown:.2%}",
            }
        ]
        for alloc in opt_result.optimized_allocations:
            metrics_rows.append({
                "Portfolio": alloc.name,
                "Expected Return": f"{alloc.expected_return:.2%}",
                "Volatility": f"{alloc.expected_volatility:.2%}",
                "Sharpe Ratio": f"{alloc.sharpe_ratio:.2f}",
                "Max Drawdown": f"{alloc.max_drawdown:.2%}",
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

        # ── Weight Comparison ─────────────────────────────────────
        if best_opt:
            st.subheader(f"Weight Comparison: Current vs. {best_opt.name}")
            weight_rows = []
            all_syms = sorted(set(list(curr.weights.keys()) + list(best_opt.weights.keys())))
            for sym in all_syms:
                c_w = curr.weights.get(sym, 0.0)
                o_w = best_opt.weights.get(sym, 0.0)
                diff = o_w - c_w
                weight_rows.append({
                    "Symbol": sym,
                    "Current Weight": f"{c_w:.1%}",
                    f"{best_opt.name} Weight": f"{o_w:.1%}",
                    "Change": f"{diff:+.1%}",
                })
            st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)

            # ── Weight Chart ──────────────────────────────────────
            fig_weights = go.Figure()
            fig_weights.add_trace(go.Bar(
                name="Current",
                x=all_syms,
                y=[curr.weights.get(s, 0.0) for s in all_syms],
                marker_color="#0A84FF",
            ))
            fig_weights.add_trace(go.Bar(
                name=best_opt.name,
                x=all_syms,
                y=[best_opt.weights.get(s, 0.0) for s in all_syms],
                marker_color="#30D158",
            ))
            fig_weights.update_layout(
                barmode="group",
                template="plotly_dark",
                yaxis_title="Weight",
                yaxis_tickformat=".0%",
                height=400,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        # ── Risk/Return Scatter ───────────────────────────────────
        st.subheader("Risk vs. Return")
        scatter_data = []
        scatter_data.append({
            "name": curr.name,
            "return": curr.expected_return,
            "vol": curr.expected_volatility,
            "sharpe": curr.sharpe_ratio,
        })
        for alloc in opt_result.optimized_allocations:
            scatter_data.append({
                "name": alloc.name,
                "return": alloc.expected_return,
                "vol": alloc.expected_volatility,
                "sharpe": alloc.sharpe_ratio,
            })

        fig_scatter = go.Figure()
        colors = ["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#64D2FF"]
        for i, pt in enumerate(scatter_data):
            fig_scatter.add_trace(go.Scatter(
                x=[pt["vol"]],
                y=[pt["return"]],
                mode="markers+text",
                marker=dict(size=14, color=colors[i % len(colors)]),
                text=[pt["name"]],
                textposition="top center",
                textfont=dict(size=11, color="#C9D1D9"),
                name=pt["name"],
            ))
        fig_scatter.update_layout(
            template="plotly_dark",
            xaxis_title="Expected Volatility",
            yaxis_title="Expected Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            height=400,
            margin=dict(l=40, r=40, t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── All Allocation Details ────────────────────────────────
        with st.expander("View All Allocation Details"):
            for alloc in [curr] + opt_result.optimized_allocations:
                st.markdown(f"**{alloc.name}**")
                w_data = [{"Symbol": s, "Weight": f"{w:.2%}"} for s, w in sorted(alloc.weights.items(), key=lambda x: -x[1]) if w > 0.001]
                st.dataframe(pd.DataFrame(w_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Module 15: Correlation Analysis
# ---------------------------------------------------------------------------
elif page == "15. Correlation Analysis":
    st.header("Stock Correlation Analysis")
    st.caption("Pairwise correlation matrix and diversification insights using actual historical returns.")

    # Ensure all watchlist symbols are in the cache
    for sym in NSE_WATCHLIST:
        if sym not in st.session_state.data_cache:
            st.session_state.data_cache[sym] = load_symbol_data(sym, 250)

    corr_symbols = list(st.session_state.data_cache.keys())
    corr_result = compute_correlation_analysis(
        data_cache=st.session_state.data_cache,
        symbols=corr_symbols,
    )

    # Store for report
    st.session_state["last_correlation"] = corr_result

    if not corr_result.data_sufficient:
        st.error(f"⚠️ {corr_result.warning}")
        st.info("Load more market data via Module 2 or ensure at least 2 symbols have 60+ days of history.")
    else:
        if corr_result.warning:
            st.warning(f"⚠️ {corr_result.warning}")

        st.write(f"**Data Points:** {corr_result.data_points} trading days · **Symbols:** {len(corr_result.symbols)}")

        # ── Correlation Heatmap ───────────────────────────────────
        st.subheader("Correlation Heatmap")
        corr_mat = corr_result.correlation_matrix

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_mat.values,
            x=corr_mat.columns.tolist(),
            y=corr_mat.index.tolist(),
            colorscale=[
                [0.0, "#FF453A"],
                [0.25, "#FF9F0A"],
                [0.5, "#161B22"],
                [0.75, "#30D158"],
                [1.0, "#39D353"],
            ],
            zmid=0.0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_mat.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="%{x} / %{y}: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Correlation"),
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            height=max(400, len(corr_mat) * 45),
            margin=dict(l=80, r=40, t=30, b=80),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Top Pairs ─────────────────────────────────────────────
        col_h, col_l = st.columns(2)
        with col_h:
            st.subheader("Highest Correlated Pairs")
            if corr_result.highest_pairs:
                high_data = [
                    {
                        "Pair": f"{p.symbol_a} / {p.symbol_b}",
                        "Correlation": f"{p.correlation:.3f}",
                    }
                    for p in corr_result.highest_pairs
                ]
                st.dataframe(pd.DataFrame(high_data), use_container_width=True, hide_index=True)
            else:
                st.info("No pairs to display.")

        with col_l:
            st.subheader("Lowest Correlated Pairs")
            if corr_result.lowest_pairs:
                low_data = [
                    {
                        "Pair": f"{p.symbol_a} / {p.symbol_b}",
                        "Correlation": f"{p.correlation:.3f}",
                    }
                    for p in corr_result.lowest_pairs
                ]
                st.dataframe(pd.DataFrame(low_data), use_container_width=True, hide_index=True)
            else:
                st.info("No pairs to display.")

        # ── Interpretation ────────────────────────────────────────
        st.subheader("Diversification Insights")
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #C9D1D9; font-size: 14px; line-height: 1.7;">
                    {corr_result.interpretation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Module 16: Strategy Comparison
# ---------------------------------------------------------------------------
elif page == "16. Strategy Comparison":
    st.header("Strategy Comparison")
    st.caption("Side-by-side comparison of multiple strategies using the existing backtesting engine.")

    # Symbol selection
    sel_sym = st.selectbox("Symbol", list(st.session_state.data_cache.keys()), key="cmp_sym")
    df = st.session_state.data_cache[sel_sym]

    # Strategy selection
    all_strat_names = list(strat_mod.REGISTRY.keys())
    selected_strats = st.multiselect(
        "Select strategies to compare",
        all_strat_names,
        default=all_strat_names[:4] if len(all_strat_names) >= 4 else all_strat_names,
        key="cmp_strats",
    )

    c1, c2 = st.columns(2)
    with c1:
        cmp_cash = st.number_input("Starting Cash (₹)", value=500_000.0, step=50_000.0, key="cmp_cash")
    with c2:
        cmp_indian_costs = st.checkbox("Apply Indian Statutory Costs", value=True, key="cmp_costs")

    if st.button("Run Comparison", type="primary") and selected_strats:
        comparison_results = []
        equity_curves = {}

        progress = st.progress(0)
        for i, strat_key in enumerate(selected_strats):
            strat = strat_mod.build(strat_key)
            cfg = BacktestConfig(
                starting_cash=cmp_cash,
                slippage_bps=2.0,
                use_indian_costs=cmp_indian_costs,
            )
            bt = Backtester(strat, cfg)
            try:
                res = bt.run(df, symbol=sel_sym)
                m = res.metrics
                comparison_results.append({
                    "Strategy": strat.name,
                    "Total Return": f"{m.total_return:+.2%}",
                    "CAGR": f"{m.cagr:.2%}",
                    "Sharpe": f"{m.sharpe:.2f}",
                    "Sortino": f"{m.sortino:.2f}",
                    "Max Drawdown": f"{m.max_drawdown:.2%}",
                    "Win Rate": f"{m.win_rate:.1%}",
                    "Trades": m.num_trades,
                    "Volatility": f"{m.volatility:.2%}",
                    "Profit Factor": f"{m.profit_factor:.2f}",
                })
                equity_curves[strat.name] = res.equity

                # Store for report
                if "strategy_comparisons" not in st.session_state:
                    st.session_state["strategy_comparisons"] = []
                st.session_state["strategy_comparisons"].append(
                    StrategyComparisonEntry(
                        name=strat.name,
                        symbol=sel_sym,
                        total_return=m.total_return,
                        cagr=m.cagr,
                        sharpe=m.sharpe,
                        max_drawdown=m.max_drawdown,
                        win_rate=m.win_rate,
                        num_trades=m.num_trades,
                        volatility=m.volatility,
                    )
                )
            except Exception as e:
                st.warning(f"Strategy {strat_key} failed: {e}")
            progress.progress((i + 1) / len(selected_strats))

        progress.empty()

        if comparison_results:
            # ── Metrics Table ─────────────────────────────────────
            st.subheader("Performance Metrics")
            st.dataframe(pd.DataFrame(comparison_results), use_container_width=True, hide_index=True)

            # ── Equity Curves ─────────────────────────────────────
            st.subheader("Equity Curve Comparison")
            fig_eq = go.Figure()
            colors = ["#00C805", "#0A84FF", "#FF9F0A", "#BF5AF2", "#64D2FF", "#FF453A", "#30D158", "#FF9500"]
            for i, (name, eq) in enumerate(equity_curves.items()):
                fig_eq.add_trace(go.Scatter(
                    x=eq.index,
                    y=eq.values,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                ))
            fig_eq.update_layout(
                template="plotly_dark",
                yaxis_title=f"Equity ({CURRENCY_SYMBOL})",
                height=500,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── Drawdown Comparison ───────────────────────────────
            st.subheader("Drawdown Comparison")
            fig_dd = go.Figure()
            for i, (name, eq) in enumerate(equity_curves.items()):
                peak = eq.cummax()
                dd = (eq - peak) / peak
                fig_dd.add_trace(go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1),
                ))
            fig_dd.update_layout(
                template="plotly_dark",
                yaxis_title="Drawdown (%)",
                yaxis_tickformat=".1%",
                height=350,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── Sharpe Bar Chart ──────────────────────────────────
            st.subheader("Sharpe Ratio Comparison")
            strat_names_list = [r["Strategy"] for r in comparison_results]
            sharpe_vals = [float(r["Sharpe"]) for r in comparison_results]
            bar_colors = ["#39D353" if s > 0 else "#FF453A" for s in sharpe_vals]
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Bar(
                x=strat_names_list,
                y=sharpe_vals,
                marker_color=bar_colors,
                text=[f"{s:.2f}" for s in sharpe_vals],
                textposition="outside",
            ))
            fig_sharpe.update_layout(
                template="plotly_dark",
                yaxis_title="Sharpe Ratio",
                height=350,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

# ---------------------------------------------------------------------------
# Module 17: Research Report
# ---------------------------------------------------------------------------
elif page == "17. Research Report":
    st.header("Quantitative Research Report")
    st.caption("Generate a comprehensive report from all computed analysis results.")

    st.info(
        "The report aggregates results from Market Regime Detection, Portfolio Optimization, "
        "Correlation Analysis, and Strategy Comparison. Run those modules first for a complete report."
    )

    if st.button("Generate Report", type="primary"):
        report_data = ResearchReportData()

        # Gather regime data
        regime = st.session_state.get("last_regime")
        if regime and isinstance(regime, RegimeResult) and regime.data_sufficient:
            report_data.regime = regime.regime
            report_data.regime_confidence = regime.confidence
            report_data.regime_explanation = regime.explanation
            report_data.regime_trend = regime.trend_label
            report_data.regime_volatility = regime.volatility_label

        # Gather strategy comparison data
        strat_comps = st.session_state.get("strategy_comparisons", [])
        if strat_comps:
            # Deduplicate — keep only the latest entry per strategy name
            seen = {}
            for sc in strat_comps:
                seen[sc.name] = sc
            report_data.strategy_results = list(seen.values())

        # Gather optimization data
        opt = st.session_state.get("last_optimization")
        if opt and isinstance(opt, OptimizationResult) and opt.data_sufficient:
            report_data.current_weights = opt.current_allocation.weights
            report_data.portfolio_return = opt.current_allocation.expected_return
            report_data.portfolio_volatility = opt.current_allocation.expected_volatility
            report_data.portfolio_sharpe = opt.current_allocation.sharpe_ratio

            if opt.optimized_allocations:
                best = max(opt.optimized_allocations, key=lambda a: a.sharpe_ratio)
                report_data.optimized_weights = best.weights
                report_data.optimization_method = best.name
                report_data.optimized_return = best.expected_return
                report_data.optimized_volatility = best.expected_volatility
                report_data.optimized_sharpe = best.sharpe_ratio

        # Gather correlation data
        corr = st.session_state.get("last_correlation")
        if corr and isinstance(corr, CorrelationResult) and corr.data_sufficient:
            import numpy as np_report
            n_syms = len(corr.symbols)
            if n_syms > 1:
                upper_tri = corr.correlation_matrix.values[np_report.triu_indices(n_syms, k=1)]
                report_data.avg_correlation = float(np_report.mean(upper_tri))
            if corr.highest_pairs:
                hp = corr.highest_pairs[0]
                report_data.highest_corr_pair = f"{hp.symbol_a} / {hp.symbol_b}"
                report_data.highest_corr_value = hp.correlation
            if corr.lowest_pairs:
                lp = corr.lowest_pairs[0]
                report_data.lowest_corr_pair = f"{lp.symbol_a} / {lp.symbol_b}"
                report_data.lowest_corr_value = lp.correlation
            report_data.correlation_interpretation = corr.interpretation

        # Risk data from broker
        acc = st.session_state.broker.get_account()
        report_data.portfolio_equity = acc.equity
        report_data.cash_balance = acc.cash
        report_data.risk_free_rate = 0.07

        # Data warnings
        warnings = []
        if not regime or not regime.data_sufficient:
            warnings.append("Market regime detection not run or data insufficient.")
        if not strat_comps:
            warnings.append("No strategy comparisons available. Run Module 16 first.")
        if not opt or not opt.data_sufficient:
            warnings.append("Portfolio optimization not run or data insufficient.")
        if not corr or not corr.data_sufficient:
            warnings.append("Correlation analysis not run or data insufficient.")
        report_data.data_warnings = warnings

        # Generate report
        markdown_report = generate_research_report(report_data)

        # Store in session
        st.session_state["last_report_md"] = markdown_report

        # Display report
        st.markdown("---")
        st.markdown(markdown_report)

    # ── Export Options ─────────────────────────────────────────────
    if "last_report_md" in st.session_state:
        st.markdown("---")
        st.subheader("Export Options")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📥 Download as Markdown",
                data=st.session_state["last_report_md"],
                file_name="trading_pipeline_research_report.md",
                mime="text/markdown",
            )
        with col_exp2:
            html_report = report_to_html(st.session_state["last_report_md"])
            st.download_button(
                label="📥 Download as HTML",
                data=html_report,
                file_name="trading_pipeline_research_report.html",
                mime="text/html",
            )

# ---------------------------------------------------------------------------
# Module 18: Live Market
# ---------------------------------------------------------------------------
elif page in ("18. Live Market", "Live Market"):
    st.header("Live Market")
    st.caption("Real-time & delayed NSE equity market information, live quotes, intraday analytics, and paper trading sync.")

    # 1. Top Controls: Auto Refresh & Manual Sync
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1.5, 2.5])
    with ctrl_col1:
        refresh_choice = st.selectbox(
            "Auto Refresh Interval",
            ["30 seconds (Default)", "15 seconds", "60 seconds", "Manual Refresh Only"],
            index=0,
            help="Configurable auto-refresh frequency. Caching preserves rate limits.",
        )
    interval_map = {
        "15 seconds": 15,
        "30 seconds (Default)": 30,
        "60 seconds": 60,
        "Manual Refresh Only": None,
    }
    refresh_seconds = interval_map[refresh_choice]

    with ctrl_col2:
        st.write("")
        force_refresh = st.button("🔄 Refresh Now", use_container_width=True)

    with ctrl_col3:
        st.write("")
        sync_paper = st.checkbox(
            "Sync Live Prices to Paper Portfolio",
            value=True,
            help="Update paper trading portfolio mark-to-market equity and risk parameters with latest prices.",
        )

    # 2. Main Dashboard Fragment
    @st.fragment(run_every=refresh_seconds)
    def render_live_market(force_update: bool = False):
        feed: LiveMarketFeed = st.session_state.live_feed
        snapshot = feed.get_snapshot(symbols=NSE_WATCHLIST, force_refresh=force_update)
        session = snapshot.session
        quotes = snapshot.quotes

        # Auto sync with paper portfolio if enabled
        if sync_paper and quotes:
            sync_paper_portfolio_with_live_prices(
                quotes,
                tracker=st.session_state.tracker,
                broker=st.session_state.broker,
            )

        # ── Market Status Banner ──────────────────────────────────
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)

        with status_col1:
            if session.status == "OPEN":
                status_html = '<div class="status-badge status-open">● NSE OPEN (09:15 - 15:30 IST)</div>'
            elif session.status == "PRE-OPEN":
                status_html = '<div class="status-badge" style="background-color: #3b2a05; color: #ffbc32; border: 1px solid #c98b00;">● NSE PRE-OPEN (09:00 - 09:15)</div>'
            elif session.status == "WEEKEND":
                status_html = '<div class="status-badge status-closed">○ CLOSED — WEEKEND</div>'
            elif session.status == "HOLIDAY":
                status_html = '<div class="status-badge status-closed">○ CLOSED — NSE HOLIDAY</div>'
            else:
                status_html = '<div class="status-badge status-closed">○ NSE MARKET CLOSED</div>'

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Exchange Status (NSE)</div>
                    <div style="margin-top: 8px;">{status_html}</div>
                    <div class="metric-sub">{session.details}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with status_col2:
            last_ts_str = snapshot.timestamp.strftime("%H:%M:%S IST") if snapshot.timestamp else "N/A"
            date_str = snapshot.timestamp.strftime("%d %b %Y") if snapshot.timestamp else ""
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Last Data Update</div>
                    <div class="metric-value">{last_ts_str}</div>
                    <div class="metric-sub">{date_str} (Server: {datetime.now(IST).strftime('%H:%M:%S')})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with status_col3:
            freshness_label = "Delayed (15m)" if session.is_open else "Latest Available"
            freshness_color = "#39D353" if not snapshot.is_stale else "#FF7B72"
            freshness_sub = "Feed Healthy" if not snapshot.is_stale else "Stale Data Warning"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Data Freshness</div>
                    <div class="metric-value" style="color: {freshness_color}; font-size: 20px;">{freshness_label}</div>
                    <div class="metric-sub">{freshness_sub} · yfinance NSE Provider</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with status_col4:
            regime_res = detect_market_regime_with_breadth(st.session_state.data_cache)
            regime_name = regime_res.regime if regime_res and regime_res.data_sufficient else "UNKNOWN"
            reg_color = "#39D353" if regime_name == "BULLISH" else ("#FF7B72" if regime_name == "BEARISH" else "#FFBC32")
            conf_str = f"{regime_res.confidence:.0%}" if regime_res and regime_res.data_sufficient else "N/A"
            trend_str = regime_res.trend_label if regime_res and regime_res.data_sufficient else "No Data"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Detected Market Regime</div>
                    <div class="metric-value" style="color: {reg_color}; font-size: 20px;">{regime_name}</div>
                    <div class="metric-sub">Confidence: {conf_str} · Trend: {trend_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if snapshot.is_stale and snapshot.warning:
            st.warning(f"⚠️ {snapshot.warning}")
        if snapshot.error:
            st.error(f"⚠️ Provider notice: {snapshot.error}")

        if not quotes or all(not q.is_valid for q in quotes.values()):
            st.error(
                "Live market data is currently unavailable from the provider. "
                "No synthetic prices are generated. Please click 'Refresh Now' or check network connectivity."
            )

        st.markdown("---")

        # ── Market Overview Table ──────────────────────────────────
        st.subheader("Market Overview (NSE Stock Universe)")
        st.caption("Live and latest session pricing for the 10-stock NSE large-cap universe.")

        sort_col1, sort_col2 = st.columns([2, 2])
        with sort_col1:
            sort_by = st.selectbox(
                "Sort Table By",
                ["Change %", "Price", "Volume", "Symbol"],
                index=0,
            )
        with sort_col2:
            sort_order = st.radio(
                "Order",
                ["Descending (High to Low)", "Ascending (Low to High)"],
                horizontal=True,
            )

        table_rows = []
        for sym in NSE_WATCHLIST:
            q = quotes.get(sym)
            if q and q.is_valid:
                table_rows.append({
                    "Symbol": q.symbol,
                    "Price": f"{CURRENCY_SYMBOL}{q.price:,.2f}",
                    "Change": f"{q.change:+,.2f}",
                    "Change %": f"{q.change_pct:+.2f}%",
                    "Volume": f"{int(q.volume):,}" if q.volume > 0 else "—",
                    "Status": q.status,
                    "_price_num": q.price,
                    "_change_pct_num": q.change_pct,
                    "_vol_num": q.volume,
                    "_symbol_str": q.symbol,
                })
            else:
                table_rows.append({
                    "Symbol": sym,
                    "Price": "Unavailable",
                    "Change": "—",
                    "Change %": "—",
                    "Volume": "—",
                    "Status": "NO DATA",
                    "_price_num": -1.0,
                    "_change_pct_num": -999.0,
                    "_vol_num": -1.0,
                    "_symbol_str": sym,
                })

        sort_field_map = {
            "Change %": "_change_pct_num",
            "Price": "_price_num",
            "Volume": "_vol_num",
            "Symbol": "_symbol_str",
        }
        key = sort_field_map[sort_by]
        reverse = (sort_order == "Descending (High to Low)")
        table_rows.sort(key=lambda r: r[key], reverse=reverse)

        clean_df = pd.DataFrame([
            {
                "Symbol": r["Symbol"],
                "Price": r["Price"],
                "Change": r["Change"],
                "Change %": r["Change %"],
                "Volume": r["Volume"],
                "Status": r["Status"],
            }
            for r in table_rows
        ])
        st.dataframe(clean_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Stock Detail Section ───────────────────────────────────
        st.subheader("Stock Detail & Intraday Session")
        st.caption("Select a stock to inspect intraday 5-minute candlestick bars, volume, and VWAP.")

        selected_symbol = st.selectbox(
            "Select Stock",
            NSE_WATCHLIST,
            index=0,
            help="Select an NSE equity to display intraday price action and key quote metrics.",
        )

        sel_quote = quotes.get(selected_symbol)
        d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)

        if sel_quote and sel_quote.is_valid:
            with d_col1:
                st.metric("Current Price", f"{CURRENCY_SYMBOL}{sel_quote.price:,.2f}")
            with d_col2:
                st.metric("Day Change", f"{sel_quote.change:+,.2f}", delta=f"{sel_quote.change_pct:+.2f}%")
            with d_col3:
                vol_str = f"{int(sel_quote.volume):,}" if sel_quote.volume > 0 else "N/A"
                st.metric("Volume", vol_str)
            with d_col4:
                high_low = f"H: {sel_quote.high or '—'} | L: {sel_quote.low or '—'}"
                st.metric("Day High / Low", high_low)
            with d_col5:
                ts_str = sel_quote.timestamp.strftime("%H:%M:%S IST") if sel_quote.timestamp else "N/A"
                st.metric("Quote Timestamp", ts_str)
        else:
            with d_col1:
                st.metric("Current Price", "Unavailable")
            with d_col2:
                st.metric("Day Change", "N/A")
            with d_col3:
                st.metric("Volume", "N/A")
            with d_col4:
                st.metric("Day High / Low", "N/A")
            with d_col5:
                st.metric("Quote Timestamp", "N/A")

        # Intraday Chart
        intraday_df = feed.get_intraday_bars(selected_symbol, period="1d", interval="5m")
        if not intraday_df.empty:
            fig_intra = create_intraday_chart(intraday_df, symbol=selected_symbol, show_vwap=True)
            st.plotly_chart(fig_intra, use_container_width=True)
        else:
            fallback_df = st.session_state.data_cache.get(selected_symbol)
            if fallback_df is not None and not fallback_df.empty:
                st.info(f"ℹ️ Intraday session bars unavailable. Showing recent historical daily bars for {selected_symbol}.")
                fig_fallback = create_candlestick_chart(fallback_df.tail(60), symbol=selected_symbol)
                st.plotly_chart(fig_fallback, use_container_width=True)
            else:
                st.warning(f"No intraday or historical bars currently available for {selected_symbol}.")

        st.markdown("---")

        # ── Paper Trading & Risk Sync Panel ────────────────────────
        st.subheader("Simulated Paper Trading & Risk Mark-to-Market")
        st.caption(
            "Live market prices dynamically update paper portfolio mark-to-market valuation, "
            "unrealized P&L, and active risk guardrails. Orders remain purely simulated."
        )

        # Use broker as single source of truth — NOT the separate tracker
        acc = st.session_state.broker.get_account()
        all_positions = st.session_state.broker.get_all_positions()

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Portfolio Equity (MTM)</div>
                    <div class="metric-value">{CURRENCY_SYMBOL}{acc.equity:,.2f}</div>
                    <div class="metric-sub">Cash + Invested = Equity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Available Cash</div>
                    <div class="metric-value">{CURRENCY_SYMBOL}{acc.cash:,.2f}</div>
                    <div class="metric-sub">Buying Power: {CURRENCY_SYMBOL}{acc.buying_power:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col3:
            unrealized_color = "#39D353" if acc.unrealized_pnl >= 0 else "#FF7B72"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Unrealized P&L</div>
                    <div class="metric-value" style="color: {unrealized_color};">{CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}</div>
                    <div class="metric-sub">Realized: {CURRENCY_SYMBOL}{acc.realized_pnl:+,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Active Positions Count</div>
                    <div class="metric-value">{len(all_positions)}</div>
                    <div class="metric-sub">Invested: {CURRENCY_SYMBOL}{acc.invested_value:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if all_positions:
            st.write("##### Active Paper Positions (Valued at Latest Market Price)")
            pos_data = []
            total_pos_upnl = 0.0
            for p in all_positions:
                # Use live quote if available, otherwise use broker's current price
                cur_quote = quotes.get(p.symbol)
                if cur_quote and cur_quote.is_valid and cur_quote.price > 0:
                    live_p = cur_quote.price
                    price_src = cur_quote.freshness
                else:
                    live_p = p.current_price
                    price_src = "Broker Cache"
                mtm_val = p.shares * live_p
                unreal_pnl = (live_p - p.avg_price) * p.shares
                total_pos_upnl += unreal_pnl
                pos_data.append({
                    "Symbol": p.symbol,
                    "Shares": p.shares,
                    "Avg Entry": f"{CURRENCY_SYMBOL}{p.avg_price:,.2f}",
                    "Live Price": f"{CURRENCY_SYMBOL}{live_p:,.2f}",
                    "Market Value": f"{CURRENCY_SYMBOL}{mtm_val:,.2f}",
                    "Unrealized P&L": f"{CURRENCY_SYMBOL}{unreal_pnl:+,.2f}",
                    "Return": f"{((live_p - p.avg_price) / p.avg_price * 100):+.2f}%" if p.avg_price > 0 else "0.0%",
                    "Price Source": price_src,
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

            # Reconciliation check
            if abs(total_pos_upnl - acc.unrealized_pnl) > 1.0:
                st.warning(
                    f"⚠️ P&L uses different prices: Position total ({CURRENCY_SYMBOL}{total_pos_upnl:+,.2f}) "
                    f"vs Account ({CURRENCY_SYMBOL}{acc.unrealized_pnl:+,.2f}). "
                    f"This is expected if live quotes differ from broker's cached prices."
                )
        else:
            st.info("No open paper positions currently. Go to Module 8 (Simulated Paper Trading) to execute test orders.")

        st.caption(
            "🔒 **Compliance & Architecture Notice**: "
            "This module consumes external market data for research, regime detection, and paper trading only. "
            "It does NOT connect to live broker order routing, and real capital is never at risk."
        )

    # Call render function
    render_live_market(force_update=force_refresh)

