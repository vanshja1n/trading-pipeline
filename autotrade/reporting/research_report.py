"""Automated Research Report Generator.

Generates a comprehensive quantitative research report from actual calculated
results. The report summarizes:

1. Market regime detection
2. Strategy comparison results
3. Portfolio allocation and optimization
4. Correlation analysis
5. Risk metrics

Output format: Markdown text (convertible to HTML/PDF).

IMPORTANT: This generator ONLY includes data that has actually been computed.
It never fabricates statistics, returns, signals, or recommendations.
All reports are clearly labelled as research/paper-trading output, not financial advice.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from ..market.constants import IST


@dataclass
class StrategyComparisonEntry:
    """Summary of a single strategy's backtest performance."""

    name: str
    symbol: str
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    volatility: float


@dataclass
class ResearchReportData:
    """Aggregated inputs for report generation."""

    # Market Regime
    regime: str = ""
    regime_confidence: float = 0.0
    regime_explanation: str = ""
    regime_trend: str = ""
    regime_volatility: str = ""

    # Strategy Comparison
    strategy_results: list[StrategyComparisonEntry] = field(default_factory=list)

    # Portfolio Analysis
    current_weights: dict[str, float] = field(default_factory=dict)
    optimized_weights: dict[str, float] = field(default_factory=dict)
    optimization_method: str = ""
    portfolio_return: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_sharpe: float = 0.0
    optimized_return: float = 0.0
    optimized_volatility: float = 0.0
    optimized_sharpe: float = 0.0

    # Correlation
    avg_correlation: float = 0.0
    highest_corr_pair: str = ""
    highest_corr_value: float = 0.0
    lowest_corr_pair: str = ""
    lowest_corr_value: float = 0.0
    correlation_interpretation: str = ""

    # Risk
    max_drawdown: float = 0.0
    risk_free_rate: float = 0.07
    portfolio_equity: float = 0.0
    cash_balance: float = 0.0

    # Metadata
    generated_at: str = ""
    data_warnings: list[str] = field(default_factory=list)


def generate_research_report(data: ResearchReportData) -> str:
    """Generate a Markdown research report from actual calculated results.

    Parameters
    ----------
    data : ResearchReportData
        All the computed values to include in the report.

    Returns
    -------
    str
        Full Markdown report text.
    """
    now = datetime.now(IST)
    timestamp = now.strftime("%d %B %Y, %H:%M IST")

    lines = [
        "# Trading Pipeline by vansh — Quantitative Research Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "> **Disclaimer:** This is a research and paper-trading report generated from "
        "simulated/historical data analysis. It does not constitute financial advice, "
        "and no real capital is at risk. All metrics are based on historical "
        "performance, which does not guarantee future results.",
        "",
        "---",
        "",
    ]

    # ── Market Regime ─────────────────────────────────────────────────
    if data.regime:
        lines.extend([
            "## Market Regime",
            "",
            f"**Current Regime:** {data.regime}",
            f"**Confidence:** {data.regime_confidence:.0%}",
            f"**Trend:** {data.regime_trend}",
            f"**Volatility:** {data.regime_volatility}",
            "",
            f"> {data.regime_explanation}",
            "",
        ])

    # ── Strategy Comparison ───────────────────────────────────────────
    if data.strategy_results:
        lines.extend([
            "## Strategy Comparison",
            "",
            "| Strategy | Total Return | CAGR | Sharpe | Max DD | Win Rate | Trades | Volatility |",
            "|----------|-------------|------|--------|--------|----------|--------|------------|",
        ])
        for s in data.strategy_results:
            lines.append(
                f"| {s.name} | {s.total_return:+.2%} | {s.cagr:.2%} | "
                f"{s.sharpe:.2f} | {s.max_drawdown:.2%} | {s.win_rate:.1%} | "
                f"{s.num_trades} | {s.volatility:.2%} |"
            )
        lines.append("")

        # Best strategy
        best = max(data.strategy_results, key=lambda x: x.sharpe)
        worst = min(data.strategy_results, key=lambda x: x.sharpe)
        lines.extend([
            f"**Best risk-adjusted performer:** {best.name} "
            f"(Sharpe: {best.sharpe:.2f})",
            f"**Lowest risk-adjusted performer:** {worst.name} "
            f"(Sharpe: {worst.sharpe:.2f})",
            "",
        ])

    # ── Portfolio Analysis ────────────────────────────────────────────
    if data.current_weights or data.optimized_weights:
        lines.extend([
            "## Portfolio Analysis",
            "",
        ])

        if data.current_weights:
            lines.append("### Current Allocation")
            lines.append("")
            lines.append("| Asset | Weight |")
            lines.append("|-------|--------|")
            for sym, w in sorted(data.current_weights.items(), key=lambda x: -x[1]):
                lines.append(f"| {sym} | {w:.1%} |")
            lines.append("")
            if data.portfolio_return != 0 or data.portfolio_volatility != 0:
                lines.append(
                    f"Expected Return: {data.portfolio_return:.2%} · "
                    f"Volatility: {data.portfolio_volatility:.2%} · "
                    f"Sharpe: {data.portfolio_sharpe:.2f}"
                )
                lines.append("")

        if data.optimized_weights:
            lines.append(f"### Optimized Allocation ({data.optimization_method})")
            lines.append("")
            lines.append("| Asset | Weight |")
            lines.append("|-------|--------|")
            for sym, w in sorted(data.optimized_weights.items(), key=lambda x: -x[1]):
                if w > 0.001:
                    lines.append(f"| {sym} | {w:.1%} |")
            lines.append("")
            if data.optimized_return != 0 or data.optimized_volatility != 0:
                lines.append(
                    f"Expected Return: {data.optimized_return:.2%} · "
                    f"Volatility: {data.optimized_volatility:.2%} · "
                    f"Sharpe: {data.optimized_sharpe:.2f}"
                )
                lines.append("")

    # ── Correlation Analysis ──────────────────────────────────────────
    if data.avg_correlation != 0 or data.highest_corr_pair:
        lines.extend([
            "## Correlation Analysis",
            "",
            f"**Average pairwise correlation:** {data.avg_correlation:.3f}",
            "",
        ])
        if data.highest_corr_pair:
            lines.append(
                f"**Most correlated pair:** {data.highest_corr_pair} "
                f"({data.highest_corr_value:.3f})"
            )
        if data.lowest_corr_pair:
            lines.append(
                f"**Least correlated pair:** {data.lowest_corr_pair} "
                f"({data.lowest_corr_value:.3f})"
            )
        lines.append("")
        if data.correlation_interpretation:
            lines.append(f"> {data.correlation_interpretation}")
            lines.append("")

    # ── Risk Analysis ─────────────────────────────────────────────────
    lines.extend([
        "## Risk Analysis",
        "",
        f"- **Portfolio Equity:** ₹{data.portfolio_equity:,.2f}",
        f"- **Cash Balance:** ₹{data.cash_balance:,.2f}",
        f"- **Maximum Drawdown:** {data.max_drawdown:.2%}",
        f"- **Risk-Free Rate (Benchmark):** {data.risk_free_rate:.2%}",
        "",
    ])

    # ── Key Findings ──────────────────────────────────────────────────
    lines.extend([
        "## Key Findings",
        "",
    ])

    findings = []
    if data.regime:
        findings.append(
            f"The market is currently classified as **{data.regime}** "
            f"with {data.regime_confidence:.0%} confidence."
        )
    if data.strategy_results:
        best = max(data.strategy_results, key=lambda x: x.sharpe)
        findings.append(
            f"Among tested strategies, **{best.name}** delivered the best "
            f"risk-adjusted return (Sharpe: {best.sharpe:.2f})."
        )
    if data.avg_correlation > 0.5:
        findings.append(
            "The portfolio shows elevated average correlation, suggesting "
            "limited diversification across holdings."
        )
    elif data.avg_correlation > 0:
        findings.append(
            f"Average portfolio correlation of {data.avg_correlation:.3f} "
            "indicates moderate diversification."
        )
    if data.optimized_sharpe > data.portfolio_sharpe and data.optimization_method:
        improvement = data.optimized_sharpe - data.portfolio_sharpe
        findings.append(
            f"Portfolio optimization ({data.optimization_method}) could improve "
            f"the Sharpe ratio by {improvement:.2f}."
        )

    if not findings:
        findings.append("Run the analysis modules to populate this section.")

    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}")

    lines.append("")

    # ── Data Warnings ─────────────────────────────────────────────────
    if data.data_warnings:
        lines.extend([
            "## Data Quality Notes",
            "",
        ])
        for w in data.data_warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Report generated by Trading Pipeline by vansh — Quantitative Research Platform.*",
        "*This is a simulated/paper-trading analysis. No real trades were executed.*",
    ])

    return "\n".join(lines)


def report_to_html(markdown_text: str) -> str:
    """Convert the Markdown report to standalone HTML for export.

    Uses basic inline conversion without requiring additional dependencies.
    """
    # Simple Markdown -> HTML conversion for export
    html_lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "<title>Trading Pipeline by vansh — Research Report</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "         background: #0D1117; color: #E6EDF3; max-width: 900px; margin: 0 auto;",
        "         padding: 40px 20px; line-height: 1.6; }",
        "  h1 { color: #F0F6FC; border-bottom: 1px solid #30363D; padding-bottom: 12px; }",
        "  h2 { color: #E6EDF3; margin-top: 32px; border-bottom: 1px solid #21262D; padding-bottom: 8px; }",
        "  h3 { color: #C9D1D9; }",
        "  table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
        "  th, td { border: 1px solid #30363D; padding: 8px 12px; text-align: left; }",
        "  th { background: #161B22; color: #8B949E; font-size: 12px; text-transform: uppercase; }",
        "  td { font-size: 14px; }",
        "  blockquote { border-left: 3px solid #30363D; padding: 8px 16px; margin: 16px 0;",
        "               color: #8B949E; font-style: italic; }",
        "  code { background: #161B22; padding: 2px 6px; border-radius: 4px; font-size: 13px; }",
        "  strong { color: #F0F6FC; }",
        "  hr { border: none; border-top: 1px solid #30363D; margin: 24px 0; }",
        "  ul, ol { padding-left: 24px; }",
        "  li { margin-bottom: 4px; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    # Very simple Markdown to HTML conversion
    in_table = False
    in_list = False
    in_blockquote = False

    for line in markdown_text.split("\n"):
        stripped = line.strip()

        # Headers
        if stripped.startswith("# "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<h3>{stripped[4:]}</h3>")

        # Table
        elif stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= set("- :") for c in cells):
                continue  # Skip separator row
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            row = "".join(f"<{tag}>{_md_inline(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")

        # Blockquote
        elif stripped.startswith(">"):
            text = stripped.lstrip("> ").strip()
            html_lines.append(f"<blockquote>{_md_inline(text)}</blockquote>")

        # Horizontal rule
        elif stripped == "---":
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append("<hr>")

        # List items
        elif stripped.startswith("- "):
            text = stripped[2:]
            html_lines.append(f"<li>{_md_inline(text)}</li>")
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == ".":
            text = stripped[2:].strip()
            html_lines.append(f"<li>{_md_inline(text)}</li>")

        # Paragraph
        elif stripped:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<p>{_md_inline(stripped)}</p>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False

    if in_table:
        html_lines.append("</table>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def _md_inline(text: str) -> str:
    """Very basic inline Markdown formatting."""
    import re
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def generate_html_report(data: ResearchReportData) -> str:
    """Generate HTML research report directly from ResearchReportData."""
    md = generate_research_report(data)
    return report_to_html(md)
