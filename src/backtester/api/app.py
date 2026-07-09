"""FastAPI service exposing the backtester over HTTP.

Endpoints:
    GET  /health    -> liveness check.
    POST /backtest  -> fetch prices, run an equal-weight momentum backtest, and
                       return performance metrics, the next-period target
                       positions, and the equity curve.

Run locally:
    uv run uvicorn backtester.api.app:app --reload
then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import math
import os
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backtester.core.costs import CostModel
from backtester.core.engine import run_backtest
from backtester.data.loader import load_prices
from backtester.data.sources import CachedPriceSource, PriceSource, YFinanceSource
from backtester.signals.momentum import momentum_positions

app = FastAPI(
    title="quant-backtesting-engine",
    version="0.1.0",
    description="Vectorized momentum backtest over yfinance data (educational/research use only).",
)

_CACHE_PATH = os.environ.get("BACKTESTER_CACHE_PATH", "prices_cache.duckdb")
_source: PriceSource | None = None


def _get_source() -> PriceSource:
    """Lazily build a DuckDB-cached yfinance source shared across requests.

    Caching at the service level means repeated /backtest calls for overlapping
    date ranges don't re-hit yfinance -- faster, and resilient to its rate limits.
    Built lazily so importing this module has no filesystem side effects.
    """
    global _source
    if _source is None:
        _source = CachedPriceSource(YFinanceSource(), _CACHE_PATH)
    return _source


class BacktestRequest(BaseModel):
    """Parameters for a momentum backtest run."""

    tickers: list[str] = Field(
        ..., min_length=1, examples=[["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]]
    )
    # `date` (not `str`) so Swagger pre-fills a valid date instead of the literal
    # "string" placeholder, and pydantic rejects malformed dates with a clear 422.
    start: date = Field(date(2022, 1, 1), description="Start date (inclusive), YYYY-MM-DD.")
    end: date | None = Field(None, description="End date (exclusive); null = up to today.")
    lookback: int = Field(60, ge=1, description="Momentum lookback in trading days.")
    asx: bool = Field(False, description="True to treat tickers as ASX-listed (append .AX).")
    initial_capital: float = Field(10_000.0, gt=0)
    fixed_fee: float = Field(0.0, ge=0)
    proportional_bps: float = Field(5.0, ge=0)
    slippage_bps: float = Field(2.0, ge=0)


class BacktestResponse(BaseModel):
    """Backtest result. NaN metrics (e.g. on too-short histories) come back as null."""

    tickers: list[str]
    n_days: int
    metrics: dict[str, float | None]
    latest_positions: dict[str, float]
    equity_curve: dict[str, float]


def _json_safe(value: float) -> float | None:
    """JSON has no NaN/Infinity; map them to null so the response stays valid."""
    return value if math.isfinite(value) else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> BacktestResponse:
    try:
        prices = load_prices(
            request.tickers,
            start=request.start.isoformat(),
            end=request.end.isoformat() if request.end else None,
            asx=request.asx,
            source=_get_source(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    n_assets = prices.shape[1]
    # Equal-weight the per-asset momentum so total gross exposure is ~100%, not n_assets.
    positions = momentum_positions(prices, lookback=request.lookback) / n_assets
    cost_model = CostModel(
        fixed_fee=request.fixed_fee,
        proportional_bps=request.proportional_bps,
        slippage_bps=request.slippage_bps,
    )
    result = run_backtest(
        prices, positions, cost_model=cost_model, initial_capital=request.initial_capital
    )

    latest = positions.iloc[-1].fillna(0.0)
    equity = result.equity_curve.dropna()
    return BacktestResponse(
        tickers=list(prices.columns),
        n_days=int(prices.shape[0]),
        metrics={k: _json_safe(v) for k, v in result.summary().items()},
        latest_positions={str(k): float(v) for k, v in latest.items()},
        equity_curve={ts.strftime("%Y-%m-%d"): float(v) for ts, v in equity.items()},
    )
