"""Walk-forward analysis and out-of-sample validation engine.

Guards against overfitting and data snooping by dividing historical data into
multiple rolling or expanding training (in-sample) and test (out-of-sample) windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import pandas as pd

from ..config import BacktestConfig, RiskConfig
from ..reporting import metrics as metrics_mod
from ..strategy.base import Strategy
from .backtest import Backtester, BacktestResult


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward split."""

    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_metrics: metrics_mod.Metrics
    oos_metrics: metrics_mod.Metrics
    is_result: BacktestResult
    oos_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Overall outcome of a multi-window walk-forward validation."""

    strategy_name: str
    symbol: str
    windows: list[WalkForwardWindow]
    combined_oos_equity: pd.Series
    combined_oos_metrics: metrics_mod.Metrics
    walk_forward_efficiency: float  # OOS Annualized Return / IS Annualized Return

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Analysis: {self.strategy_name} on {self.symbol}",
            f"Number of Windows: {len(self.windows)}",
            f"Walk-Forward Efficiency (WFE): {self.walk_forward_efficiency:.2%}",
            f"Combined OOS Sharpe: {self.combined_oos_metrics.sharpe:.2f}",
            f"Combined OOS Return: {self.combined_oos_metrics.total_return:.2%}",
            f"Combined OOS Max Drawdown: {self.combined_oos_metrics.max_drawdown:.2%}",
        ]
        return "\n".join(lines)


def run_walk_forward(
    strategy_builder: Callable[[], Strategy],
    bars: pd.DataFrame,
    symbol: str = "NSE",
    num_windows: int = 4,
    train_pct: float = 0.7,
    config: BacktestConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation across chronological time splits.

    Parameters
    ----------
    strategy_builder : Callable[[], Strategy]
        Factory function returning a freshly initialized Strategy instance.
    bars : pd.DataFrame
        Full historical OHLCV data.
    symbol : str
        Asset symbol.
    num_windows : int
        Number of consecutive out-of-sample test splits.
    train_pct : float
        Proportion of each slice dedicated to In-Sample training/warmup.
    config : BacktestConfig, optional
    risk_config : RiskConfig, optional

    Returns
    -------
    WalkForwardResult
    """
    cfg = config or BacktestConfig()
    risk_cfg = risk_config or RiskConfig()

    sample_strat = strategy_builder()
    warmup = getattr(sample_strat, "warmup", 0)
    min_required = max(100, warmup + 30)
    n = len(bars)
    if n < min_required:
        raise ValueError(
            f"Need at least {min_required} bars for walk-forward validation of {sample_strat.name} "
            f"(warmup: {warmup}), got {n}"
        )

    # Calculate step size for windows
    window_size = int(n / (1 + (num_windows - 1) * (1 - train_pct)))
    step_size = int(window_size * (1 - train_pct))

    windows: list[WalkForwardWindow] = []
    oos_equity_segments: list[pd.Series] = []
    is_returns: list[float] = []
    oos_returns: list[float] = []

    last_capital = cfg.starting_cash

    for win_idx in range(num_windows):
        start_idx = win_idx * step_size
        end_idx = min(start_idx + window_size, n)

        window_bars = bars.iloc[start_idx:end_idx]
        split_idx = int(len(window_bars) * train_pct)

        train_bars = window_bars.iloc[:split_idx]
        test_bars = window_bars.iloc[split_idx:]

        if len(train_bars) < max(20, warmup + 5) or len(test_bars) < 10:
            continue

        # In-sample backtest
        is_strat = strategy_builder()
        is_bt = Backtester(is_strat, cfg, risk_cfg)
        is_res = is_bt.run(train_bars, symbol=symbol)

        # Out-of-sample backtest with warmup history from preceding train window
        oos_strat = strategy_builder()
        oos_cfg = BacktestConfig(
            starting_cash=last_capital,
            slippage_bps=cfg.slippage_bps,
            commission_per_share=cfg.commission_per_share,
            use_indian_costs=cfg.use_indian_costs,
            periods_per_year=cfg.periods_per_year,
            risk_free_rate=cfg.risk_free_rate,
        )
        oos_bt = Backtester(oos_strat, oos_cfg, risk_cfg)

        warmup_len = min(warmup, len(train_bars))
        if warmup_len > 0:
            oos_input_bars = window_bars.iloc[split_idx - warmup_len:]
            full_oos_res = oos_bt.run(oos_input_bars, symbol=symbol)
            test_idx = test_bars.index
            oos_eq = full_oos_res.equity.loc[test_idx]
            if not oos_eq.empty and oos_eq.iloc[0] > 0:
                oos_eq = (oos_eq / oos_eq.iloc[0]) * last_capital

            if not full_oos_res.trades.empty:
                test_start_ts = test_bars.index[0]
                oos_trades = full_oos_res.trades[full_oos_res.trades["timestamp"] >= test_start_ts].copy()
            else:
                oos_trades = full_oos_res.trades

            oos_positions = full_oos_res.positions.loc[test_idx]
            oos_weights = full_oos_res.target_weights.loc[test_idx]
            oos_metrics = metrics_mod.compute(
                oos_eq,
                oos_trades,
                oos_weights,
                periods_per_year=cfg.periods_per_year,
                risk_free_rate=cfg.risk_free_rate,
            )
            oos_res = BacktestResult(
                equity=oos_eq,
                positions=oos_positions,
                target_weights=oos_weights,
                trades=oos_trades,
                metrics=oos_metrics,
                strategy_name=oos_strat.name,
                symbol=symbol,
                halted=full_oos_res.halted,
                halt_reason=full_oos_res.halt_reason,
                _config=oos_cfg,
            )
        else:
            oos_res = oos_bt.run(test_bars, symbol=symbol)

        last_capital = float(oos_res.equity.iloc[-1])
        oos_equity_segments.append(oos_res.equity)

        is_returns.append(is_res.metrics.annualized_return)
        oos_returns.append(oos_res.metrics.annualized_return)

        windows.append(
            WalkForwardWindow(
                window_index=win_idx + 1,
                train_start=train_bars.index[0],
                train_end=train_bars.index[-1],
                test_start=test_bars.index[0],
                test_end=test_bars.index[-1],
                is_metrics=is_res.metrics,
                oos_metrics=oos_res.metrics,
                is_result=is_res,
                oos_result=oos_res,
            )
        )

    # Stitch OOS equity
    if oos_equity_segments:
        combined_equity = pd.concat(oos_equity_segments)
        combined_equity = combined_equity[~combined_equity.index.duplicated(keep="first")]
    else:
        combined_equity = pd.Series([cfg.starting_cash], index=[bars.index[0]])

    mean_is_ret = sum(is_returns) / len(is_returns) if is_returns else 0.0
    mean_oos_ret = sum(oos_returns) / len(oos_returns) if oos_returns else 0.0
    wfe = (mean_oos_ret / mean_is_ret) if mean_is_ret > 0 else 0.0

    combined_metrics = metrics_mod.compute(
        combined_equity,
        pd.DataFrame(),
        pd.Series(1.0, index=combined_equity.index),
        periods_per_year=cfg.periods_per_year,
        risk_free_rate=cfg.risk_free_rate,
    )

    first_strat = strategy_builder()

    return WalkForwardResult(
        strategy_name=first_strat.name,
        symbol=symbol,
        windows=windows,
        combined_oos_equity=combined_equity,
        combined_oos_metrics=combined_metrics,
        walk_forward_efficiency=wfe,
    )
