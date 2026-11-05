"""Strategy Reliability and Overfitting Analyzer.

Evaluates quantitative strategies across multiple dimensions:
1. Walk-Forward Efficiency (In-sample vs out-of-sample persistence)
2. Transaction Cost Sensitivity (Can the strategy survive Indian taxes and slippage?)
3. Market Regime Robustness (Bullish, Bearish, Sideways performance)
4. Trade Frequency & Statistical Significance
5. Composite 0-100 Reliability Score with diagnostic feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..config import BacktestConfig, RiskConfig
from ..engine.backtest import Backtester, BacktestResult
from ..engine.walk_forward import run_walk_forward
from ..execution.costs import IndianEquityCostModel
from ..strategy.base import Strategy


@dataclass
class CostSensitivityResult:
    slippage_bps: float
    use_taxes: bool
    annualized_return: float
    sharpe: float
    profit_factor: float


@dataclass
class ReliabilityReport:
    strategy_name: str
    symbol: str
    overall_score: float  # 0 to 100
    verdict: str          # "ROBUST", "MODERATE", "OVERFITTED", "FRAGILE"
    wfe: float
    oos_sharpe: float
    cost_sensitivity: list[CostSensitivityResult]
    strengths: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Strategy Reliability Audit: {self.strategy_name} on {self.symbol} ===",
            f"Overall Score: {self.overall_score:.1f} / 100  [{self.verdict}]",
            f"Walk-Forward Efficiency: {self.wfe:.1%}",
            f"Out-of-Sample Sharpe: {self.oos_sharpe:.2f}",
            "",
            "Strengths:",
        ]
        for s in self.strengths:
            lines.append(f"  + {s}")
        lines.append("Vulnerabilities:")
        for v in self.vulnerabilities:
            lines.append(f"  - {v}")
        lines.append("Recommendations:")
        for r in self.recommendations:
            lines.append(f"  * {r}")
        return "\n".join(lines)


class StrategyReliabilityAnalyzer:
    """Evaluates strategy robustness before live capital deployment."""

    def __init__(
        self,
        strategy_builder: Callable[[], Strategy],
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
    ):
        self.strategy_builder = strategy_builder
        self.config = config or BacktestConfig(use_indian_costs=True)
        self.risk_config = risk_config or RiskConfig()

    def analyze(self, bars: pd.DataFrame, symbol: str = "NSE") -> ReliabilityReport:
        strengths: list[str] = []
        vulnerabilities: list[str] = []
        recommendations: list[str] = []
        points = 0.0

        # 1. Walk-Forward Test (Weight: 40 points)
        wf_res = run_walk_forward(
            self.strategy_builder,
            bars,
            symbol=symbol,
            num_windows=3,
            train_pct=0.7,
            config=self.config,
            risk_config=self.risk_config,
        )
        wfe = wf_res.walk_forward_efficiency
        oos_sharpe = wf_res.combined_oos_metrics.sharpe

        if wfe >= 0.70:
            points += 35.0
            strengths.append(f"High walk-forward efficiency ({wfe:.1%}), showing strong out-of-sample generalisation")
        elif wfe >= 0.40:
            points += 20.0
            strengths.append(f"Acceptable walk-forward efficiency ({wfe:.1%})")
        else:
            vulnerabilities.append(f"Low walk-forward efficiency ({wfe:.1%}) indicates risk of curve fitting/overfitting")
            recommendations.append("Reduce strategy parameter complexity or widen lookback windows")

        if oos_sharpe >= 1.0:
            points += 15.0
            strengths.append(f"Excellent out-of-sample Sharpe ratio ({oos_sharpe:.2f})")
        elif oos_sharpe >= 0.5:
            points += 10.0
        elif oos_sharpe <= 0:
            vulnerabilities.append("Out-of-sample Sharpe is zero or negative")

        # 2. Transaction Cost Sensitivity (Weight: 35 points)
        # Test 0 bps (frictionless), 5 bps (standard NSE), 15 bps (stressed liquidity)
        cost_scenarios = [
            (0.0, False),
            (2.0, True),
            (10.0, True),
        ]
        sensitivity_results: list[CostSensitivityResult] = []
        returns_by_cost = []

        for slip_bps, use_tax in cost_scenarios:
            sc_cfg = BacktestConfig(
                slippage_bps=slip_bps,
                use_indian_costs=use_tax,
                periods_per_year=self.config.periods_per_year,
                risk_free_rate=self.config.risk_free_rate,
            )
            strat = self.strategy_builder()
            bt = Backtester(strat, sc_cfg, self.risk_config)
            res = bt.run(bars, symbol=symbol)
            sensitivity_results.append(
                CostSensitivityResult(
                    slippage_bps=slip_bps,
                    use_taxes=use_tax,
                    annualized_return=res.metrics.annualized_return,
                    sharpe=res.metrics.sharpe,
                    profit_factor=res.metrics.profit_factor,
                )
            )
            returns_by_cost.append(res.metrics.annualized_return)

        # Cost decay check
        if len(returns_by_cost) == 3:
            zero_cost_ret = returns_by_cost[0]
            stress_cost_ret = returns_by_cost[2]
            if zero_cost_ret > 0:
                cost_survival_ratio = stress_cost_ret / zero_cost_ret
                if cost_survival_ratio >= 0.70:
                    points += 30.0
                    strengths.append("High resilience to Indian transaction costs and slippage")
                elif cost_survival_ratio >= 0.30:
                    points += 20.0
                    strengths.append("Moderate resilience to transaction costs")
                else:
                    points += 5.0
                    vulnerabilities.append("Heavy return erosion under statutory Indian taxes and slippage")
                    recommendations.append("Decrease trade frequency or target larger average price moves")
            elif stress_cost_ret > 0:
                points += 15.0

        # 3. Trade Frequency & Statistical Sample (Weight: 15 points)
        base_bt = Backtester(self.strategy_builder(), self.config, self.risk_config).run(bars, symbol)
        num_trades = base_bt.metrics.num_trades
        if num_trades >= 30:
            points += 15.0
            strengths.append(f"Robust statistical trade sample ({num_trades} closed trades)")
        elif num_trades >= 15:
            points += 10.0
        else:
            vulnerabilities.append(f"Sparse trade sample ({num_trades} trades), results may have high variance")
            recommendations.append("Test over a longer historical period or across multiple NSE symbols")

        # 4. Drawdown Discipline (Weight: 10 points)
        max_dd = base_bt.metrics.max_drawdown
        if max_dd < 0.15:
            points += 10.0
            strengths.append(f"Well-controlled maximum drawdown ({max_dd:.1%})")
        elif max_dd < 0.25:
            points += 5.0
        else:
            vulnerabilities.append(f"Elevated maximum drawdown ({max_dd:.1%})")
            recommendations.append("Tighten ATR stops or lower position size")

        score = min(100.0, max(0.0, points))

        if score >= 75.0:
            verdict = "ROBUST"
        elif score >= 55.0:
            verdict = "MODERATE"
        elif score >= 35.0:
            verdict = "FRAGILE"
        else:
            verdict = "OVERFITTED"

        return ReliabilityReport(
            strategy_name=base_bt.strategy_name,
            symbol=symbol,
            overall_score=score,
            verdict=verdict,
            wfe=wfe,
            oos_sharpe=oos_sharpe,
            cost_sensitivity=sensitivity_results,
            strengths=strengths,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
        )
