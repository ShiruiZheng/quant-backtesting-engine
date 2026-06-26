"""Price data sources behind a small interface, so the rest of the code never
depends on *where* prices come from.

`PriceSource` is the contract (a Protocol): "give me closes for these tickers
over this date range, as a date-indexed DataFrame with one column per ticker".
Two implementations:

* `YFinanceSource` -- fetches live from yfinance (one call per ticker, so one
  bad/delisted ticker is skipped with a warning instead of failing the batch).
* `CachedPriceSource` -- wraps *another* source and a DuckDB file: it serves
  from the local store when the requested range is already cached, and only
  calls the wrapped source for the missing span. This decouples the app from
  yfinance at request time (faster, fewer rate-limit failures, reproducible).

This is the dependency-inversion seam discussed in [[decisions.md]]: high-level
code (`load_prices`, the engine, the API) depends on the `PriceSource` abstraction,
not on yfinance or DuckDB directly. Sources return *raw* closes (NaNs where a
ticker has no quote); alignment/forward-fill is applied once by
`backtester.data.loader.load_prices`, so caching stores range-independent data.
"""

from __future__ import annotations

import os
import warnings
from datetime import date
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import yfinance as yf


@runtime_checkable
class PriceSource(Protocol):
    """Anything that can return closes for `tickers` over `[start, end]`."""

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        *,
        price_field: str = "Close",
    ) -> pd.DataFrame:
        """Return a date-indexed DataFrame of `price_field`, one column per ticker.

        Values are raw (no forward-fill); dates a ticker has no quote for are NaN.
        Raises ValueError if none of the requested tickers returned any data.
        """
        ...


class YFinanceSource:
    """`PriceSource` backed by live yfinance downloads."""

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        *,
        price_field: str = "Close",
    ) -> pd.DataFrame:
        series: dict[str, pd.Series] = {}
        for ticker in tickers:
            history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if history.empty or price_field not in history:
                warnings.warn(f"No data returned for ticker '{ticker}'; skipping.", stacklevel=2)
                continue
            series[ticker] = history[price_field]

        if not series:
            raise ValueError(f"No price data returned for any of tickers={tickers!r}")

        prices = pd.concat(series, axis=1)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        return prices


class CachedPriceSource:
    """Wrap a `PriceSource` with a DuckDB-backed local cache.

    Coverage is tracked per (ticker, field) as the single contiguous span that
    has actually been fetched. A request inside that span is served from the
    store; a request that extends it triggers one fetch of the *union* span
    (which includes any gap, so the store never claims data it doesn't hold).
    """

    def __init__(self, source: PriceSource, cache_path: str | os.PathLike[str]) -> None:
        self._source = source
        self._cache_path = str(cache_path)
        self._con: Any = None  # lazily opened so construction has no side effects

    def _connect(self) -> Any:
        if self._con is None:
            import duckdb

            parent = os.path.dirname(self._cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._con = duckdb.connect(self._cache_path)
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS prices "
                "(ticker VARCHAR, field VARCHAR, day DATE, close DOUBLE, "
                "PRIMARY KEY (ticker, field, day))"
            )
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS coverage "
                "(ticker VARCHAR, field VARCHAR, span_start DATE, span_end DATE, "
                "PRIMARY KEY (ticker, field))"
            )
        return self._con

    def _coverage(self, ticker: str, field: str) -> tuple[date, date] | None:
        row = self._connect().execute(
            "SELECT span_start, span_end FROM coverage WHERE ticker = ? AND field = ?",
            [ticker, field],
        ).fetchone()
        return (row[0], row[1]) if row else None

    def _ensure_cached(self, ticker: str, start: date, end: date, field: str) -> None:
        cov = self._coverage(ticker, field)
        if cov is not None and cov[0] <= start and cov[1] >= end:
            return  # requested range already cached

        span_start = min(start, cov[0]) if cov else start
        span_end = max(end, cov[1]) if cov else end
        try:
            frame = self._source.get_prices(
                [ticker], span_start.isoformat(), span_end.isoformat(), price_field=field
            )
        except ValueError:
            return  # upstream has nothing for this ticker; leave the cache untouched

        if frame.shape[1] == 0:
            return
        col = frame[ticker] if ticker in frame.columns else frame.iloc[:, 0]

        con = self._connect()
        con.execute(
            "DELETE FROM prices WHERE ticker = ? AND field = ? AND day BETWEEN ? AND ?",
            [ticker, field, span_start, span_end],
        )
        rows = [
            (ticker, field, ts.date() if isinstance(ts, pd.Timestamp) else ts, float(value))
            for ts, value in col.dropna().items()
        ]
        if rows:
            con.executemany("INSERT INTO prices VALUES (?, ?, ?, ?)", rows)
        con.execute("DELETE FROM coverage WHERE ticker = ? AND field = ?", [ticker, field])
        con.execute(
            "INSERT INTO coverage VALUES (?, ?, ?, ?)",
            [ticker, field, span_start, span_end],
        )

    def _read_cached(self, ticker: str, start: date, end: date, field: str) -> pd.Series:
        rows = self._connect().execute(
            "SELECT day, close FROM prices "
            "WHERE ticker = ? AND field = ? AND day BETWEEN ? AND ? ORDER BY day",
            [ticker, field, start, end],
        ).fetchall()
        if not rows:
            return pd.Series(dtype=float, name=ticker)
        index = pd.to_datetime([r[0] for r in rows])
        return pd.Series([r[1] for r in rows], index=index, name=ticker)

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        *,
        price_field: str = "Close",
    ) -> pd.DataFrame:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end) if end else date.today()

        series: dict[str, pd.Series] = {}
        for ticker in tickers:
            self._ensure_cached(ticker, start_d, end_d, price_field)
            cached = self._read_cached(ticker, start_d, end_d, price_field)
            if not cached.empty:
                series[ticker] = cached

        if not series:
            raise ValueError(f"No price data returned for any of tickers={tickers!r}")
        return pd.concat(series, axis=1)
