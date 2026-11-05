"""SQLite persistence layer for market data, trading signals, orders, and portfolio state.

Designed with clean schemas and parameterized queries that are compatible with
both SQLite and PostgreSQL.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Sequence

import pandas as pd


class Database:
    """SQLite database manager for quantitative trading persistence."""

    def __init__(self, db_path: str | Path = "data/database/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Create platform tables if they do not exist."""
        with self._connection() as conn:
            cursor = conn.cursor()
            # 1. Market Data Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_sym_time ON market_data(symbol, timestamp)"
            )

            # 2. Signals Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    signal REAL NOT NULL,
                    metadata TEXT
                )
                """
            )

            # 3. Orders & Executions Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    costs REAL NOT NULL,
                    notes TEXT
                )
                """
            )

            # 4. Trades Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    costs REAL NOT NULL
                )
                """
            )

            # 5. Portfolio Snapshots Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    positions_json TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------ Market Data
    def save_market_data(self, symbol: str, df: pd.DataFrame) -> int:
        """Upsert OHLCV data for a given symbol.

        Returns number of rows written.
        """
        if df.empty:
            return 0

        rows = []
        for ts, row in df.iterrows():
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append(
                (
                    symbol.upper(),
                    ts_str,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
            )

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timestamp) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume
                """,
                rows,
            )
        return len(rows)

    def load_market_data(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load stored OHLCV data into a canonical DataFrame."""
        query = "SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = ?"
        params: list[Any] = [symbol.upper()]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp ASC"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        from ..market.constants import IST
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
        df = df.set_index("timestamp")
        df.attrs["symbol"] = symbol.upper()
        return df

    def get_available_symbols(self) -> list[str]:
        """Return list of distinct symbols stored in database."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_data ORDER BY symbol ASC")
            return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------ Signals
    def log_signal(
        self,
        timestamp: datetime | str,
        symbol: str,
        strategy_name: str,
        signal: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a trading signal."""
        ts_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        meta_str = json.dumps(metadata or {})
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO signals (timestamp, symbol, strategy_name, signal, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts_str, symbol.upper(), strategy_name, float(signal), meta_str),
            )

    # ------------------------------------------------------------ Orders
    def log_order(
        self,
        order_id: str,
        timestamp: datetime | str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_type: str = "market",
        status: str = "filled",
        costs: float = 0.0,
        notes: str = "",
    ) -> None:
        """Record an order execution."""
        ts_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO orders (order_id, timestamp, symbol, side, qty, price, order_type, status, costs, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    ts_str,
                    symbol.upper(),
                    side.upper(),
                    float(qty),
                    float(price),
                    order_type,
                    status,
                    float(costs),
                    notes,
                ),
            )

    def load_orders(self, symbol: str | None = None) -> pd.DataFrame:
        """Load executed orders."""
        query = "SELECT * FROM orders"
        params: list[Any] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY timestamp DESC"
        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ------------------------------------------------------------ Trades
    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        strategy_name: str,
        entry_time: str,
        exit_time: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl: float,
        pnl_pct: float,
        costs: float = 0.0,
    ) -> None:
        """Record a closed trade."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trades (trade_id, symbol, strategy_name, entry_time, exit_time, entry_price, exit_price, qty, pnl, pnl_pct, costs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    symbol.upper(),
                    strategy_name,
                    entry_time,
                    exit_time,
                    float(entry_price),
                    float(exit_price),
                    float(qty),
                    float(pnl),
                    float(pnl_pct),
                    float(costs),
                ),
            )

    def load_trades(self, symbol: str | None = None) -> pd.DataFrame:
        """Load closed trades."""
        query = "SELECT * FROM trades"
        params: list[Any] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY exit_time DESC"
        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
