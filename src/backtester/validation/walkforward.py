"""Walk-forward (rolling out-of-sample) validation harness.

A single train/test split flatters a strategy: whatever parameter or hedge
ratio you chose was chosen knowing that one test period. Walk-forward instead
rolls a (train, test) window forward through time -- fit / choose on the
trailing `train_size` window, evaluate on the next `test_size` window, step
forward, repeat. Concatenating the test windows yields one continuous
out-of-sample return stream, the closest offline proxy for "how it would have
behaved if actually deployed".

The harness is signal-agnostic. You pass a `strategy` callable mapping
``(train_prices, test_prices) -> positions`` aligned to the test index. The
contract:

* The harness guarantees the *window split* -- the strategy only sees `train`
  before it must produce positions for `test`. Any parameter fitting (a hedge
  ratio, a chosen lookback) must use `train` only.
* The strategy is responsible for being lookahead-safe *within* the test window
  (the signal functions in `backtester.signals` already are). It may read
  `train` purely as warm-up history so the first test-day signal isn't NaN --
  that is causal and allowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from backtester.core.costs import CostModel
from backtester.core.engine import run_backtest
from backtester.core.metrics import TRADING_DAYS_PER_YEAR, summarize

Strategy = Callable[[pd.DataFrame, pd.DataFrame], "pd.DataFrame | pd.Series"]


@dataclass(frozen=True)
class WalkForwardResult:
    """Stitched out-of-sample results across all rolling windows."""

    oos_returns: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]
    window_metrics: list[dict[str, object]] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.window_metrics)


def walk_forward(
    prices: pd.DataFrame | pd.Series,
    strategy: Strategy,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    cost_model: CostModel | None = None,
    initial_capital: float = 10_000.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> WalkForwardResult:
    """Roll (train, test) windows forward and aggregate the out-of-sample returns.

    `step` defaults to `test_size`, giving back-to-back, non-overlapping test
    windows. Each test window is simulated independently (it starts flat), so a
    small amount of turnover is charged at every window boundary -- a deliberate,
    conservative simplification.
    """
    prices_df = prices.to_frame() if isinstance(prices, pd.Series) else prices
    if cost_model is None:
        cost_model = CostModel()
    step = step or test_size

    n = len(prices_df)
    if train_size + test_size > n:
        raise ValueError(
            f"Not enough data: need train_size + test_size = {train_size + test_size} "
            f"rows, have {n}."
        )

    oos_chunks: list[pd.Series] = []
    window_metrics: list[dict[str, object]] = []

    start = 0
    while start + train_size + test_size <= n:
        train = prices_df.iloc[start : start + train_size]
        test = prices_df.iloc[start + train_size : start + train_size + test_size]

        positions = strategy(train, test)
        result = run_backtest(
            test,
            positions,
            cost_model=cost_model,
            initial_capital=initial_capital,
            periods_per_year=periods_per_year,
        )
        oos_chunks.append(result.returns)

        window: dict[str, object] = dict(
            summarize(result.returns, periods_per_year=periods_per_year)
        )
        window["train_start"] = str(train.index[0])
        window["train_end"] = str(train.index[-1])
        window["test_start"] = str(test.index[0])
        window["test_end"] = str(test.index[-1])
        window_metrics.append(window)

        start += step

    oos_returns = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float)
    equity_curve = initial_capital * (1.0 + oos_returns).cumprod()
    metrics = summarize(oos_returns, periods_per_year=periods_per_year)

    return WalkForwardResult(
        oos_returns=oos_returns,
        equity_curve=equity_curve,
        metrics=metrics,
        window_metrics=window_metrics,
    )
