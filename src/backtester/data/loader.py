"""Price loading: ticker-name handling + alignment, on top of a `PriceSource`.

This module owns the two concerns that are independent of *where* prices come
from: (1) resolving exchange-specific ticker names (the ASX `.AX` suffix), and
(2) aligning/cleaning the result. Fetching itself is delegated to a
`PriceSource` (see `backtester.data.sources`), so `load_prices` works the same
whether prices come live from yfinance or from the DuckDB cache.

Forward-fill is applied here, once, after fetching: a forward-fill only ever
copies a *past* value forward, so it carries no lookahead risk (unlike
interpolation, which could pull a future value into a past gap).
"""

from __future__ import annotations

import pandas as pd

from backtester.data.sources import PriceSource, YFinanceSource


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
    source: PriceSource | None = None,
) -> pd.DataFrame:
    """Download price history for one or more tickers into a single aligned DataFrame.

    Returns a DataFrame indexed by date with one column per ticker, holding
    `price_field` (default "Close", auto-adjusted for splits/dividends). Tickers
    that return no data are skipped with a warning; dates where every requested
    ticker is missing are dropped, and remaining gaps are forward-filled.

    `source` selects where prices come from and defaults to live yfinance
    (`YFinanceSource`). Pass a `CachedPriceSource` to read/write the DuckDB cache.

    Raises ValueError if none of the requested tickers returned any data.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if asx:
        tickers = [to_asx_ticker(t) for t in tickers]

    if source is None:
        source = YFinanceSource()

    prices = source.get_prices(tickers, start, end, price_field=price_field)
    return prices.dropna(how="all").ffill()
