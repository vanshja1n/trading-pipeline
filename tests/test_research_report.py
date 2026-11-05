"""Tests for Automated Research Report Generator."""

import pytest

from autotrade.reporting.research_report import (
    ResearchReportData,
    StrategyComparisonEntry,
    generate_research_report,
    generate_html_report,
)


def test_research_report_generation():
    data = ResearchReportData(
        regime="BULLISH",
        regime_confidence=0.85,
        regime_explanation="Price is above 50-day and 200-day SMAs with positive momentum.",
        regime_trend="Strong Uptrend",
        regime_volatility="Normal Volatility",
        strategy_results=[
            StrategyComparisonEntry(
                name="EMA Crossover",
                symbol="RELIANCE",
                total_return=0.182,
                cagr=0.145,
                sharpe=1.25,
                max_drawdown=-0.082,
                win_rate=0.58,
                num_trades=24,
                volatility=0.16,
            ),
            StrategyComparisonEntry(
                name="RSI Mean Reversion",
                symbol="RELIANCE",
                total_return=0.095,
                cagr=0.078,
                sharpe=0.82,
                max_drawdown=-0.115,
                win_rate=0.52,
                num_trades=32,
                volatility=0.14,
            ),
        ],
        optimization_method="Maximum Sharpe Ratio",
        current_weights={"RELIANCE": 0.45, "TCS": 0.35, "INFY": 0.20},
        optimized_weights={"RELIANCE": 0.50, "TCS": 0.30, "INFY": 0.20},
        portfolio_return=0.155,
        portfolio_volatility=0.128,
        portfolio_sharpe=1.21,
        avg_correlation=0.42,
        highest_corr_pair="TCS / INFY",
        highest_corr_value=0.74,
        lowest_corr_pair="RELIANCE / INFY",
        lowest_corr_value=0.21,
        correlation_interpretation="Moderate cross-asset diversification.",
    )

    md = generate_research_report(data)
    assert "# Trading Pipeline by vansh — Quantitative Research Report" in md
    assert "Disclaimer:" in md
    assert "RELIANCE" in md
    assert "EMA Crossover" in md
    assert "Maximum Sharpe Ratio" in md
    assert "RELIANCE" in md

    html = generate_html_report(data)
    assert "<!DOCTYPE html>" in html
    assert "Trading Pipeline" in html
    assert "Quantitative Research Report" in html
    assert "Disclaimer" in html
