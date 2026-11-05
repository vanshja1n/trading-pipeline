"""Tests for Indian transaction costs, SimulatedBroker, PortfolioTracker, and Reliability Analyzer."""

from datetime import datetime
import pandas as pd
import numpy as np
import pytest

from autotrade.execution.costs import IndianEquityCostModel, FlatCostModel
from autotrade.execution.simulated import SimulatedBroker
from autotrade.portfolio.tracker import PortfolioTracker
from autotrade.reliability.analyzer import StrategyReliabilityAnalyzer
from autotrade.strategy.strategies import SMACrossover, BuyAndHold


def test_indian_cost_model_delivery():
    cost_model = IndianEquityCostModel()

    # Buy 100 shares @ ₹2,500 = ₹2,50,000 turnover
    buy_costs = cost_model.calculate(side="BUY", shares=100, price=2500.0, is_intraday=False)
    assert buy_costs.turnover == 250_000.0
    # STT: 0.1% = ₹250
    assert buy_costs.stt == 250.0
    # Stamp duty: 0.015% = ₹37.5
    assert buy_costs.stamp_duty == 37.5
    # Total costs > STT + Stamp duty
    assert buy_costs.total > 287.5

    # Sell 100 shares @ ₹2,600 = ₹2,60,000 turnover
    sell_costs = cost_model.calculate(side="SELL", shares=100, price=2600.0, is_intraday=False)
    # Stamp duty on delivery sell is 0
    assert sell_costs.stamp_duty == 0.0
    # STT on delivery sell is 0.1% = ₹260
    assert sell_costs.stt == 260.0


def test_simulated_broker_lifecycle():
    broker = SimulatedBroker(starting_cash=100_000.0)
    broker.update_price("RELIANCE", 2500.0)

    # Check initial account
    acc = broker.get_account()
    assert acc.cash == 100_000.0
    assert acc.equity == 100_000.0

    # Buy 10 shares of RELIANCE
    order = broker.submit_order("RELIANCE", 10, "buy")
    assert order.status == "filled"
    assert broker.get_position("RELIANCE") == 10

    # Price moves to 2600
    broker.update_price("RELIANCE", 2600.0)
    acc = broker.get_account()
    # Equity should reflect unrealized profit minus costs
    assert acc.equity > 100_500.0

    # Close position
    close_order = broker.close_position("RELIANCE")
    assert close_order is not None
    assert broker.get_position("RELIANCE") == 0


def test_portfolio_tracker():
    tracker = PortfolioTracker(starting_cash=200_000.0)
    # Buy 20 shares of TCS at 3500 (cost: 70,000)
    tracker.record_fill("TCS", "buy", 20, 3500.0, costs=50.0)
    snap = tracker.get_snapshot()

    assert snap.total_equity == pytest.approx(199_950.0, rel=1e-3)
    assert snap.allocations["TCS"] > 0.30
    assert snap.allocations["CASH"] < 0.70


def test_reliability_analyzer():
    # Create 120 bars of synthetic upward trending price data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="B", tz="Asia/Kolkata")
    drift = np.linspace(100, 150, 120)
    noise = np.random.normal(0, 1.0, 120)
    close = drift + noise
    high = close + 1.5
    low = close - 1.5
    open_ = close - 0.5
    volume = np.random.uniform(50000, 100000, 120)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    analyzer = StrategyReliabilityAnalyzer(lambda: BuyAndHold())
    report = analyzer.analyze(df, symbol="RELIANCE")

    assert report.overall_score >= 0.0
    assert report.verdict in ("ROBUST", "MODERATE", "OVERFITTED", "FRAGILE")
    assert len(report.cost_sensitivity) == 3
