from __future__ import annotations

import pandas as pd
import pytest

from backtester.data import sources
from backtester.data.loader import load_prices, to_asx_ticker


class FakeTicker:
    """Stand-in for yf.Ticker that returns canned history without network access."""

    _histories: dict[str, pd.DataFrame] = {}

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, start: str, end: str | None = None, auto_adjust: bool = True) -> pd.DataFrame:
        return self._histories.get(self.symbol, pd.DataFrame())


@pytest.fixture(autouse=True)
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    # yfinance now lives in the YFinanceSource (backtester.data.sources); load_prices
    # delegates to it, so patching here makes the default source return canned data.
    monkeypatch.setattr(sources.yf, "Ticker", FakeTicker)


def _set_history(symbol: str, dates: list[str], closes: list[float]) -> None:
    FakeTicker._histories[symbol] = pd.DataFrame(
        {"Close": closes}, index=pd.DatetimeIndex(dates, name="Date")
    )


@pytest.fixture(autouse=True)
def clear_histories() -> None:
    FakeTicker._histories = {}


def test_to_asx_ticker_appends_suffix() -> None:
    assert to_asx_ticker("bhp") == "BHP.AX"


def test_to_asx_ticker_is_idempotent() -> None:
    assert to_asx_ticker("BHP.AX") == "BHP.AX"
    assert to_asx_ticker("bhp.ax") == "BHP.AX"


def test_load_prices_single_ticker() -> None:
    _set_history("BHP.AX", ["2024-01-01", "2024-01-02", "2024-01-03"], [40.0, 41.0, 42.0])
    prices = load_prices("BHP", start="2024-01-01")
    assert list(prices.columns) == ["BHP.AX"]
    assert prices.shape[0] == 3
    assert prices["BHP.AX"].tolist() == [40.0, 41.0, 42.0]


def test_load_prices_multiple_tickers_aligned() -> None:
    _set_history("BHP.AX", ["2024-01-01", "2024-01-02", "2024-01-03"], [40.0, 41.0, 42.0])
    _set_history("CBA.AX", ["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 99.0])
    prices = load_prices(["BHP", "CBA"], start="2024-01-01")
    assert list(prices.columns) == ["BHP.AX", "CBA.AX"]
    assert prices.shape == (3, 2)


def test_load_prices_forward_fills_gaps() -> None:
    _set_history("BHP.AX", ["2024-01-01", "2024-01-02", "2024-01-03"], [40.0, 41.0, 42.0])
    _set_history("CBA.AX", ["2024-01-01", "2024-01-03"], [100.0, 99.0])
    prices = load_prices(["BHP", "CBA"], start="2024-01-01")
    # CBA has no quote for 2024-01-02; it should be forward-filled from 2024-01-01, not NaN.
    assert prices["CBA.AX"].tolist() == [100.0, 100.0, 99.0]


def test_load_prices_skips_ticker_with_no_data_and_warns() -> None:
    _set_history("BHP.AX", ["2024-01-01", "2024-01-02"], [40.0, 41.0])
    # CBA.AX intentionally has no history set -> FakeTicker returns empty DataFrame.
    with pytest.warns(UserWarning, match="CBA.AX"):
        prices = load_prices(["BHP", "CBA"], start="2024-01-01")
    assert list(prices.columns) == ["BHP.AX"]


def test_load_prices_raises_if_all_tickers_empty() -> None:
    with pytest.raises(ValueError, match="No price data"):
        load_prices(["ZZZ"], start="2024-01-01")


def test_load_prices_asx_false_does_not_append_suffix() -> None:
    _set_history("AAPL", ["2024-01-01", "2024-01-02"], [190.0, 191.0])
    prices = load_prices("AAPL", start="2024-01-01", asx=False)
    assert list(prices.columns) == ["AAPL"]
