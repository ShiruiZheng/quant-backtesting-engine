import numpy as np
import pandas as pd
import pytest

from backtester.api import app as api


@pytest.fixture
def fake_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=80)
    rng = np.random.default_rng(0)
    data = {
        t: 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=80)))
        for t in ("AAPL", "MSFT", "GOOGL")
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture(autouse=True)
def patch_loader(monkeypatch: pytest.MonkeyPatch, fake_prices: pd.DataFrame) -> None:
    # Never hit yfinance from a unit test; serve canned prices instead.
    monkeypatch.setattr(api, "load_prices", lambda *a, **k: fake_prices)


def test_health() -> None:
    assert api.health() == {"status": "ok"}


def test_backtest_returns_metrics_and_positions() -> None:
    request = api.BacktestRequest(tickers=["AAPL", "MSFT", "GOOGL"], lookback=5)
    response = api.backtest(request)

    assert response.tickers == ["AAPL", "MSFT", "GOOGL"]
    assert response.n_days == 80
    assert "sharpe_ratio" in response.metrics
    assert set(response.latest_positions) == {"AAPL", "MSFT", "GOOGL"}
    assert len(response.equity_curve) > 0


def test_backtest_maps_loader_failure_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ValueError("No price data returned")

    monkeypatch.setattr(api, "load_prices", boom)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        api.backtest(api.BacktestRequest(tickers=["ZZZ"]))
    assert excinfo.value.status_code == 400


def test_request_rejects_non_date_string() -> None:
    # The Swagger "Try it out" placeholder "string" must not slip through as a date.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        api.BacktestRequest(tickers=["AAPL"], end="string")


def test_request_parses_iso_date_strings() -> None:
    from datetime import date

    request = api.BacktestRequest(tickers=["AAPL"], start="2022-01-01", end="2023-06-30")
    assert request.start == date(2022, 1, 1)
    assert request.end == date(2023, 6, 30)


def test_backtest_passes_iso_dates_to_loader(
    monkeypatch: pytest.MonkeyPatch, fake_prices: pd.DataFrame
) -> None:
    captured: dict[str, object] = {}

    def capturing_loader(tickers, start, end=None, asx=False):  # type: ignore[no-untyped-def]
        captured["start"] = start
        captured["end"] = end
        return fake_prices

    monkeypatch.setattr(api, "load_prices", capturing_loader)
    api.backtest(
        api.BacktestRequest(tickers=["AAPL", "MSFT", "GOOGL"], start="2022-01-01", lookback=5)
    )
    # date objects are converted to ISO strings; null end stays None (= up to today).
    assert captured["start"] == "2022-01-01"
    assert captured["end"] is None
