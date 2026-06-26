"""Price data loading for ASX equities via yfinance.

This module only fetches and aligns raw prices; it does not compute returns
or signals, so it carries no lookahead risk of its own. yfinance can return
gaps for thinly-traded ASX tickers or temporary delistings -- those are
forward-filled here since a forward-fill only ever uses past information.
"""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf


def to_asx_ticker(symbol: str) -> str:
    """Append the `.AX` suffix yfinance expects for ASX-listed tickers, if missing."""
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(".AX") else f"{symbol}.AX"


def load_prices(
    tickers: str | list[str],
    start: str,
    end: str | None = None,
    *,
    price_field: str = "Close",
    asx: bool = True,
) -> pd.DataFrame:
    """Download price history for one or more tickers into a single aligned DataFrame.

    Returns a DataFrame indexed by date with one column per ticker, holding
    `price_field` (default "Close", which yfinance auto-adjusts for splits and
    dividends). Tickers that return no data are skipped with a warning rather
    than failing the whole request. Dates where every requested ticker is
    missing are dropped; remaining gaps are forward-filled.

    Raises ValueError if none of the requested tickers returned any data.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if asx:
        tickers = [to_asx_ticker(t) for t in tickers]

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

    prices = prices.dropna(how="all").ffill()
    return prices
