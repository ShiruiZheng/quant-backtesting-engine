"""Transaction cost model: fixed fee, proportional fee, slippage, and market impact.

A trade is any change in position size between consecutive periods. Costs are
computed vectorized over the whole position-change series (no Python loops),
so the model can be applied directly to a positions DataFrame/Series produced
by a signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Per-trade cost parameters.

    fixed_fee: flat currency cost charged on any nonzero trade (e.g. brokerage).
    proportional_bps: linear fee on traded notional, in basis points (1 bps = 1e-4).
    slippage_bps: linear cost on traded notional representing bid/ask + execution
        slippage, in basis points.
    market_impact_coef: coefficient for a simplified quadratic-in-size impact cost,
        impact = market_impact_coef * traded_size^2 * price. This approximates the
        intuition that larger trades move the price against you disproportionately
        more; it is a research simplification, not a calibrated market-impact model.
    """

    fixed_fee: float = 0.0
    proportional_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_coef: float = 0.0

    def trade_costs(
        self,
        position_changes: pd.DataFrame | pd.Series,
        prices: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame | pd.Series:
        """Total transaction cost incurred at each date, in currency units.

        `position_changes` is the period-over-period change in position size
        (e.g. `positions.diff()`); `prices` must be aligned to the same index.
        """
        traded_size = position_changes.abs()
        notional = traded_size * prices

        fixed = (traded_size > 0).astype(float) * self.fixed_fee
        proportional = notional * (self.proportional_bps / 1e4)
        slippage = notional * (self.slippage_bps / 1e4)
        market_impact = self.market_impact_coef * (traded_size**2) * prices

        return fixed + proportional + slippage + market_impact

    def trade_cost_fraction(
        self,
        weight_changes: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame | pd.Series:
        """Transaction cost as a *fraction of capital*, for weight-based backtests.

        The vectorized equity-curve engine works in portfolio weights, not share
        counts: a trade of ``|delta_weight|`` moves that fraction of capital, and
        the linear costs apply directly to it. So this returns
        ``|delta_weight| * (proportional_bps + slippage_bps) / 1e4``.

        Only the two linear (bps) components appear here -- they are the ones with
        a natural fraction-of-notional meaning. `fixed_fee` (a flat currency
        amount) and `market_impact` (currency, price-dependent) are share/notional
        concepts and live in `trade_costs`; mixing them into weight space would
        require an assumed capital base and per-share price, which the engine does
        not carry. See [[decisions.md]] for the reasoning.
        """
        return weight_changes.abs() * ((self.proportional_bps + self.slippage_bps) / 1e4)
