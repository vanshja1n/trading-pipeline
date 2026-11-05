#!/usr/bin/env python
"""Run the bot against an Alpaca paper account.

Requires ALPACA_API_KEY / ALPACA_API_SECRET in .env (free at alpaca.markets --
create *paper* keys).

    python run_paper.py --check                       # verify the connection only
    python run_paper.py --symbols SPY QQQ             # dry run: logs orders, sends none
    python run_paper.py --symbols SPY --execute       # actually submit paper orders

Live money is intentionally awkward to reach: you need ALPACA_LIVE=true in the
environment AND --i-know-this-is-real-money on the command line. Don't do that
until a paper run has behaved for weeks.
"""

from __future__ import annotations

import argparse
import logging
import sys

from autotrade.broker.alpaca import AlpacaBroker, DryRunBroker
from autotrade.config import LiveConfig, RiskConfig
from autotrade.data.feed import AlpacaFeed
from autotrade.engine.live import AlreadyRunning, LiveTrader
from autotrade.strategy import REGISTRY, build


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["SPY"])
    p.add_argument("--strategy", default="sma_crossover", choices=sorted(REGISTRY))
    p.add_argument("--timeframe", default="1Day",
                   choices=["1Min", "5Min", "15Min", "1Hour", "1Day"])
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--lookback", type=int, default=400, help="bars fetched per cycle")
    p.add_argument("--rebalance-threshold", type=float, default=0.05,
                   help="minimum weight drift before rebalancing; suppresses churn")
    p.add_argument("--max-drawdown", type=float, default=0.15)
    p.add_argument("--max-weight", type=float, default=0.95)
    p.add_argument("--stop-atr", type=float, default=0.0)
    p.add_argument("--execute", action="store_true", help="submit orders instead of dry-running")
    p.add_argument("--check", action="store_true", help="connect, print account, exit")
    p.add_argument("--once", action="store_true", help="run a single decision cycle and exit")
    p.add_argument("--i-know-this-is-real-money", dest="allow_live", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = LiveConfig(
        symbols=[s.upper() for s in args.symbols],
        timeframe=args.timeframe,
        lookback_bars=args.lookback,
        poll_seconds=args.poll_seconds,
        rebalance_threshold=args.rebalance_threshold,
        dry_run=not args.execute,
        allow_live_money=args.allow_live,
    )

    if not config.api_key or not config.api_secret:
        print("ERROR: no Alpaca credentials found.\n"
              "  1. Sign up free at https://alpaca.markets and generate PAPER keys\n"
              "  2. cp .env.example .env  and paste them in", file=sys.stderr)
        return 1

    broker = AlpacaBroker(config.api_key, config.api_secret, paper=config.paper)
    account = broker.get_account()
    print(f"\nConnected to {'PAPER' if config.paper else 'LIVE'} account")
    print(f"  equity        ${account.equity:,.2f}")
    print(f"  cash          ${account.cash:,.2f}")
    print(f"  buying power  ${account.buying_power:,.2f}")
    print(f"  market open   {broker.is_market_open()}\n")

    if args.check:
        return 0

    if config.dry_run:
        broker = DryRunBroker(broker)
        print("DRY RUN: orders will be logged, not submitted. Add --execute when ready.\n")

    trader = LiveTrader(
        strategy=build(args.strategy),
        broker=broker,
        feed=AlpacaFeed(config.api_key, config.api_secret),
        config=config,
        risk_config=RiskConfig(
            max_position_weight=args.max_weight,
            max_drawdown=args.max_drawdown,
            stop_loss_atr=args.stop_atr,
        ),
    )
    try:
        trader.run(max_iterations=1 if args.once else None)
    except AlreadyRunning as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
