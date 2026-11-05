#!/usr/bin/env python
"""Backtest a strategy on historical data.

Examples:
    python run_backtest.py --symbol SPY --strategy sma_crossover --start 2015-01-01
    python run_backtest.py --symbol QQQ --strategy rsi_reversion --stop-atr 2.0
    python run_backtest.py --symbol AAPL --compare
"""

from __future__ import annotations

import argparse
import logging
import sys

from autotrade.config import BacktestConfig, RiskConfig
from autotrade.data.feed import YahooFeed
from autotrade.engine.backtest import Backtester
from autotrade.strategy import REGISTRY, build


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="RELIANCE")
    p.add_argument("--strategy", default="sma_crossover", choices=sorted(REGISTRY))
    p.add_argument("--market", default="IN", choices=["IN", "US"], help="market context (IN or US)")
    p.add_argument("--timeframe", default="1Day",
                   choices=["1Min", "5Min", "15Min", "1Hour", "1Day"],
                   help="bar size; must match what you plan to trade live")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cash", type=float, default=500_000.0)
    p.add_argument("--slippage-bps", type=float, default=2.0, help="cost per fill in basis points")
    p.add_argument("--indian-costs", action="store_true", default=True, help="model Indian STT, GST, and exchange fees")
    p.add_argument("--max-drawdown", type=float, default=0.20, help="kill switch level, e.g. 0.20")
    p.add_argument("--max-daily-loss", type=float, default=0.05,
                   help="flatten after this loss in a single day")
    p.add_argument("--max-weight", type=float, default=0.95, help="max fraction of equity per position")
    p.add_argument("--stop-atr", type=float, default=0.0, help="ATR stop multiple; 0 disables")
    p.add_argument("--target-vol", type=float, default=0.0, help="annualized vol target; 0 disables")
    p.add_argument("--compare", action="store_true", help="run every registered strategy")
    p.add_argument("--csv", default=None, help="write the equity curve to this path")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Sharpe/CAGR annualize by bar count, so this MUST track the bar size or
    # every risk-adjusted number in the report is silently wrong.
    bars_per_year = {
        "1Day": 252,
        "1Hour": 252 * 7,      # ~7 hourly bars in a US equity session
        "15Min": 252 * 26,
        "5Min": 252 * 78,
        "1Min": 252 * 390,
    }[args.timeframe]

    bt_config = BacktestConfig(
        starting_cash=args.cash,
        slippage_bps=args.slippage_bps,
        periods_per_year=bars_per_year,
        use_indian_costs=args.indian_costs and (args.market == "IN"),
    )
    risk_config = RiskConfig(
        max_position_weight=args.max_weight,
        max_drawdown=args.max_drawdown,
        max_daily_loss=args.max_daily_loss,
        stop_loss_atr=args.stop_atr,
        target_volatility=args.target_vol,
    )

    if args.timeframe != "1Day":
        # Yahoo only keeps a short window of intraday history: ~60 days for
        # minute bars, ~730 for hourly. Ask for more and you get nothing back.
        limit = "7 days" if args.timeframe == "1Min" else (
            "60 days" if args.timeframe in ("5Min", "15Min") else "730 days")
        print(f"\nNote: Yahoo serves only about the last {limit} of {args.timeframe} bars.")
        print("Short history means a short backtest -- treat the result as a smoke")
        print("test, not evidence. Alpaca's data goes back further if you need it.")

    bars = YahooFeed(market=args.market).history(args.symbol, args.start, args.end, timeframe=args.timeframe)
    print(f"\nLoaded {len(bars)} bars for {args.symbol} "
          f"({bars.index[0].date()} to {bars.index[-1].date()})\n")

    names = sorted(REGISTRY) if args.compare else [args.strategy]
    results = []
    for name in names:
        strategy = build(name)
        if len(bars) <= strategy.warmup:
            print(f"skipping {name}: needs more than {strategy.warmup} bars")
            continue
        result = Backtester(strategy, bt_config, risk_config).run(bars, args.symbol)
        results.append(result)
        print(result.summary())
        print()

    if args.compare and results:
        print("-" * 70)
        print(f"{'strategy':<20}{'return':>10}{'CAGR':>10}{'sharpe':>9}{'max DD':>10}{'trades':>8}")
        print("-" * 70)
        for r in sorted(results, key=lambda r: r.metrics.sharpe, reverse=True):
            m = r.metrics
            print(f"{r.strategy_name:<20}{m.total_return:>9.1%}{m.cagr:>10.1%}"
                  f"{m.sharpe:>9.2f}{m.max_drawdown:>10.1%}{m.num_trades:>8d}"
                  f"{'  HALTED' if r.halted else ''}")
        print("-" * 70)
        print("Sharpe-ranked. If nothing beats BuyAndHold, that's the honest answer.")
        if any(r.halted for r in results):
            print("HALTED = the kill switch fired and the strategy sat in cash for the rest\n"
                  "of the run (resuming is manual, by design). Re-run with --max-drawdown 0.99\n"
                  "--max-daily-loss 0.99 to see raw strategy performance without it.")
        print()

    if args.csv and results:
        results[0].equity.to_csv(args.csv)
        print(f"equity curve written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
