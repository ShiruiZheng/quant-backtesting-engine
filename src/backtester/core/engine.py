"""Vectorized portfolio / PnL backtest engine.

This is the piece that turns *positions* into an *equity curve*: it combines
each position with the return realized over the period it is held, subtracts
transaction costs, and compounds the result into a growing (or shrinking)
account value. No Python loops -- the whole simulation is a handful of
vectorized pandas operations over the aligned price/position frames.

Convention (critical -- read before using):

    positions.loc[t] is the position *held during* period t.

That position must have been decided using only information available before
period t began. The signal functions in `backtester.signals` already apply the
mandatory `.shift(1)`, so their output drops straight in here and this engine
does NOT shift again (doing so would double-lag and silently distort results).
The return earned over period t is the simple return ``p_t / p_{t-1} - 1``, so
the contribution of an asset at t is ``position_t * return_t``.

Positions are interpreted as portfolio *weights* (fraction of capital per
asset). Costs are charged on the change in weight (turnover) each period via
`CostModel.trade_cost_fraction`, including the initial move off zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.core.costs import CostModel
from backtester.core.metrics import TRADING_DAYS_PER_YEAR, max_drawdown, summarize
from backtester.core.returns import simple_returns


def _as_frame(obj: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Treat a single-asset Series as a one-column frame so the math is uniform."""
    return obj.to_frame() if isinstance(obj, pd.Series) else obj


@dataclass(frozen=True)
class BacktestResult:
    """Everything produced by a single backtest run.

    `returns` is the net (after-cost) per-period portfolio return -- the series
    every metric in `backtester.core.metrics` consumes.
    """

    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame
    initial_capital: float
    periods_per_year: int

    @property
    def final_equity(self) -> float:
        if self.equity_curve.empty:
            return self.initial_capital
        return float(self.equity_curve.iloc[-1])

    @property
    def max_drawdown(self) -> float:
        return max_drawdown(self.returns)

    def summary(self) -> dict[str, float]:
        """Standard metric bundle plus engine-only stats (turnover, final equity)."""
        stats = summarize(self.returns, periods_per_year=self.periods_per_year)
        stats["avg_turnover"] = float(self.turnover.mean()) if not self.turnover.empty else 0.0
        stats["final_equity"] = self.final_equity
        return stats


def run_backtest(
    prices: pd.DataFrame | pd.Series,
    positions: pd.DataFrame | pd.Series,
    *,
    cost_model: CostModel | None = None,
    initial_capital: float = 10_000.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Simulate holding `positions` against `prices` into an equity curve.

    `positions` are weights already lagged to be tradable (see module docstring).
    They are aligned to `prices` (index and columns); missing entries are treated
    as flat. `cost_model` defaults to a zero-cost model.
    """
    if cost_model is None:
        cost_model = CostModel()

    prices_df = _as_frame(prices)
    held = (
        _as_frame(positions)
        .reindex(index=prices_df.index, columns=prices_df.columns)
        .astype(float)
        .fillna(0.0)
    )
    asset_returns = simple_returns(prices_df).fillna(0.0)

    gross_returns = (held * asset_returns).sum(axis=1)

    weight_changes = held.diff()
    if not weight_changes.empty:
        # The first period establishes the position from flat, which is a real trade.
        weight_changes.iloc[0] = held.iloc[0]
    turnover = weight_changes.abs().sum(axis=1)
    costs = cost_model.trade_cost_fraction(weight_changes).sum(axis=1)

    net_returns = gross_returns - costs
    equity_curve = initial_capital * (1.0 + net_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        turnover=turnover,
        positions=held,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
    )
