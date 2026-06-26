"""End-to-end demo of the full backtesting pipeline against live US equity data.

This fetches real prices from yfinance and walks through every piece of the
engine, in order:

  1. Load prices for a basket of US stocks.
  2. Momentum signal -> the next-day target positions ("what to hold tomorrow").
  3. Vectorized backtest -> equity curve + performance/risk metrics.
  4. Walk-forward validation -> honest out-of-sample metrics (not in-sample).
  5. Engle-Granger pairs / stat-arb on two of the names, with its own backtest.

It needs network access (it hits yfinance every run; there is no cache layer).

Run with:
    uv run python scripts/run_demo.py
    uv run python scripts/run_demo.py --tickers AAPL MSFT NVDA --start 2021-01-01 --lookback 40
    uv run python scripts/run_demo.py --asx --tickers BHP CBA CSL    # ASX instead of US
"""

from __future__ import annotations

import argparse

import pandas as pd

from backtester.core.costs import CostModel
from backtester.core.engine import run_backtest
from backtester.data.loader import load_prices
from backtester.data.sources import CachedPriceSource, PriceSource, YFinanceSource
from backtester.signals.momentum import momentum_positions
from backtester.signals.pairs import engle_granger, pairs_positions
from backtester.validation.walkforward import walk_forward

# Five large, liquid US names by default ("top 5 for next day" -> see section 2).
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    for name, value in metrics.items():
        print(f"  {name:24s} {value:>12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument(
        "--asx", action="store_true", help="Treat tickers as ASX-listed (append .AX)."
    )
    parser.add_argument(
        "--cache",
        default=".cache/prices.duckdb",
        help="DuckDB cache path; re-runs read from here instead of re-downloading.",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass the cache and always hit yfinance."
    )
    args = parser.parse_args()

    cost_model = CostModel(fixed_fee=0.0, proportional_bps=5.0, slippage_bps=2.0)

    # 1. Load prices (cached unless --no-cache) ------------------------------
    source: PriceSource = YFinanceSource()
    if not args.no_cache:
        source = CachedPriceSource(YFinanceSource(), args.cache)
        print(f"Using price cache at {args.cache} (re-runs of the same range are instant).")
    prices = load_prices(args.tickers, start=args.start, end=args.end, asx=args.asx, source=source)
    n_assets = prices.shape[1]
    print(f"Loaded {prices.shape[0]} rows x {n_assets} tickers: {list(prices.columns)}")
    print("\nLast 3 rows of prices:")
    print(prices.tail(3))

    # 2. Momentum signal -> next-day target positions ------------------------
    positions = momentum_positions(prices, lookback=args.lookback) / n_assets
    next_day = positions.iloc[-1].fillna(0.0)
    print(f"\nNext-day target weights (momentum, lookback={args.lookback}):")
    for ticker, weight in next_day.items():
        side = "LONG" if weight > 0 else "SHORT" if weight < 0 else "flat"
        print(f"  {ticker:10s} {weight:+.3f}  ({side})")

    # 3. Full backtest -> equity curve + metrics -----------------------------
    result = run_backtest(prices, positions, cost_model=cost_model, initial_capital=10_000.0)
    print(f"\nFinal equity from $10,000: ${result.final_equity:,.2f}")
    _print_metrics("In-sample backtest metrics:", result.summary())

    # 4. Walk-forward validation (out-of-sample) -----------------------------
    def momentum_strategy(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        # Use train purely as warm-up history so the first test-day signal is not NaN.
        full = pd.concat([train, test])
        return momentum_positions(full, lookback=args.lookback).reindex(test.index) / n_assets

    train_size = min(252, max(args.lookback + 5, len(prices) // 3))
    test_size = max(21, len(prices) // 10)
    if train_size + test_size <= len(prices):
        wf = walk_forward(
            prices,
            momentum_strategy,
            train_size=train_size,
            test_size=test_size,
            cost_model=cost_model,
        )
        _print_metrics(
            f"Walk-forward OOS metrics ({wf.n_windows} windows, "
            f"train={train_size}/test={test_size}):",
            wf.metrics,
        )
    else:
        print("\n(Not enough history for walk-forward; fetch a longer --start range.)")

    # 5. Engle-Granger pairs / stat-arb --------------------------------------
    if n_assets >= 2:
        a, b = prices.columns[0], prices.columns[1]
        coint = engle_granger(prices[a], prices[b])
        print(
            f"\nEngle-Granger {a} ~ {b}: hedge_ratio={coint.hedge_ratio:.3f}, "
            f"ADF p-value={coint.adf_pvalue:.4f} "
            f"({'cointegrated' if coint.is_cointegrated() else 'not cointegrated'} at 5%)"
        )
        pair_pos = pairs_positions(prices[a], prices[b], lookback=20, entry_z=2.0, exit_z=0.5)
        pair_result = run_backtest(prices[[a, b]], pair_pos, cost_model=cost_model)
        _print_metrics(f"Pairs backtest metrics ({a}/{b}):", pair_result.summary())

    print("\nDone. Try the HTTP API too:  uv run uvicorn backtester.api.app:app --reload")


if __name__ == "__main__":
    main()
