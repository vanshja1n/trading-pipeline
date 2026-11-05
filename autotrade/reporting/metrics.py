"""Performance statistics for an equity curve.

Total return is the least useful number here. Look at max drawdown and Sharpe
first: they tell you whether the return was worth what you went through.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    num_trades: int
    exposure: float
    final_equity: float

    @property
    def annualized_return(self) -> float:
        """Alias for CAGR."""
        return self.cagr

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        rows = [
            ("Final equity", f"${self.final_equity:,.2f}"),
            ("Total return", f"{self.total_return:>8.2%}"),
            ("CAGR", f"{self.cagr:>8.2%}"),
            ("Volatility (ann.)", f"{self.volatility:>8.2%}"),
            ("Sharpe", f"{self.sharpe:>8.2f}"),
            ("Sortino", f"{self.sortino:>8.2f}"),
            ("Max drawdown", f"{self.max_drawdown:>8.2%}"),
            ("Calmar", f"{self.calmar:>8.2f}"),
            ("Win rate", f"{self.win_rate:>8.2%}"),
            ("Profit factor", f"{self.profit_factor:>8.2f}"),
            ("Trades", f"{self.num_trades:>8d}"),
            ("Time in market", f"{self.exposure:>8.2%}"),
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(f"  {label:<{width}}  {value}" for label, value in rows)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def compute(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    exposure: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Metrics:
    equity = equity.dropna()
    if len(equity) < 2:
        raise ValueError("need at least two equity points to compute metrics")

    returns = equity.pct_change().dropna()
    periods = len(returns)
    years = periods / periods_per_year

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    vol = returns.std() * np.sqrt(periods_per_year)
    excess = returns - risk_free_rate / periods_per_year
    sharpe = (excess.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0.0

    downside = returns[returns < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else 0.0

    max_dd = abs(drawdown_series(equity).min())
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    win_rate = profit_factor = 0.0
    num_trades = 0
    if trades is not None and not trades.empty and "pnl" in trades.columns:
        closed = trades[trades["pnl"].notna()]
        num_trades = len(closed)
        if num_trades:
            wins = closed[closed["pnl"] > 0]["pnl"]
            losses = closed[closed["pnl"] < 0]["pnl"]
            win_rate = len(wins) / num_trades
            profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")

    time_in_market = float((exposure.abs() > 0).mean()) if exposure is not None and len(exposure) else 0.0

    return Metrics(
        total_return=float(total_return),
        cagr=float(cagr),
        volatility=float(vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        calmar=float(calmar),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        num_trades=int(num_trades),
        exposure=time_in_market,
        final_equity=float(equity.iloc[-1]),
    )
