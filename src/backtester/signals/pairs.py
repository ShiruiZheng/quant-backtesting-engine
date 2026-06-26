"""Engle-Granger cointegration and a pairs / statistical-arbitrage signal.

Engle-Granger tests whether two price series share a long-run equilibrium: fit
``y = a + b*x`` by OLS, then test the residual spread ``y - (a + b*x)`` for
stationarity with an Augmented Dickey-Fuller (ADF) test. A low ADF p-value is
evidence the spread mean-reverts -- the bet a pairs trade makes.

Trading rule: standardize the spread into a rolling z-score and trade its
reversion. Go long the spread (long ``y``, short ``b`` units of ``x``) when the
z-score is unusually low, short it when unusually high, and flatten when it
reverts toward zero. As everywhere in this codebase, the returned positions are
shifted by one period (`.shift(1)`) so the position held during ``t`` depends
only on the z-score known by the close of ``t-1`` -- no lookahead.

Important caveat: `engle_granger` fit on the *whole* sample is in-sample. The
hedge ratio it picks has "seen" the entire price path, so a backtest using it
directly is optimistic. For an honest test, re-estimate the hedge ratio on a
trailing window only -- pass it via `hedge_ratio=` and drive the rolling
re-fit with `backtester.validation.walk_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class CointegrationResult:
    """Output of an Engle-Granger fit on two price series."""

    hedge_ratio: float
    intercept: float
    adf_pvalue: float
    spread: pd.Series

    def is_cointegrated(self, alpha: float = 0.05) -> bool:
        """True if the spread's ADF p-value clears the significance level `alpha`."""
        return self.adf_pvalue < alpha


def engle_granger(y: pd.Series, x: pd.Series) -> CointegrationResult:
    """Fit ``y = intercept + hedge_ratio * x`` and ADF-test the residual spread.

    Only overlapping, non-NaN observations are used. Raises ValueError if the
    two series do not overlap.
    """
    df = pd.concat({"y": y, "x": x}, axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping non-NaN observations between y and x.")

    exog = sm.add_constant(df["x"])
    fit = sm.OLS(df["y"], exog).fit()
    intercept = float(fit.params["const"])
    hedge_ratio = float(fit.params["x"])
    spread = df["y"] - (intercept + hedge_ratio * df["x"])
    adf_pvalue = float(adfuller(spread)[1])
    return CointegrationResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        adf_pvalue=adf_pvalue,
        spread=spread,
    )


def spread_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """Rolling z-score of the spread over a trailing `lookback` window.

    Uses a trailing window only (so it is causal); the consumer still shifts the
    derived position by one period before trading on it.
    """
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std


def pairs_positions(
    y: pd.Series,
    x: pd.Series,
    *,
    lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    hedge_ratio: float | None = None,
) -> pd.DataFrame:
    """Lagged long/short weights for a mean-reversion pairs trade on (`y`, `x`).

    Enter when ``|z| >= entry_z`` (fade the move), exit when ``|z| <= exit_z``,
    and hold in between. Returns a two-column DataFrame of weights for the `y`
    and `x` legs, already shifted one period so it is safe to trade.

    If `hedge_ratio` is None it is estimated in-sample via `engle_granger`
    (convenient but optimistic -- see the module docstring). Pass an explicit
    value (e.g. one fit on a trailing window) for an out-of-sample test.
    """
    df = pd.concat({"y": y, "x": x}, axis=1)
    if hedge_ratio is None:
        hedge_ratio = engle_granger(df["y"], df["x"]).hedge_ratio

    spread = df["y"] - hedge_ratio * df["x"]
    z = spread_zscore(spread, lookback=lookback)

    # Stateful entry/exit with hysteresis, expressed vectorially: mark entries and
    # exits, leave everything else NaN, then forward-fill to "hold" the position.
    target = pd.Series(np.nan, index=z.index)
    target[z >= entry_z] = -1.0  # spread rich -> short the spread
    target[z <= -entry_z] = 1.0  # spread cheap -> long the spread
    target[z.abs() <= exit_z] = 0.0  # reverted -> flat
    spread_position = target.ffill().fillna(0.0)

    y_name = y.name if y.name is not None else "y"
    x_name = x.name if x.name is not None else "x"
    positions = pd.DataFrame(
        {
            y_name: spread_position,
            x_name: -hedge_ratio * spread_position,
        },
        index=z.index,
    )
    return positions.shift(1)
