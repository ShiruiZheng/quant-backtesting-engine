"""Momentum signal generation.

A position held *during* period t can only be informed by data known by the
close of period t-1 -- otherwise the backtest is trading on information it
would not have had in real time (lookahead bias). `momentum_score` is the
trailing return as of each date and is NOT yet safe to trade on. `momentum_signal`
and `momentum_positions` apply the mandatory one-period shift before the value
is used as a trading input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_score(
    prices: pd.DataFrame | pd.Series, lookback: int = 60
) -> pd.DataFrame | pd.Series:
    """Trailing total return over `lookback` periods, as known at each date.

    score(t) = p_t / p_{t-lookback} - 1. This uses prices up to and including
    t, so it must be shifted before being used to decide the position held
    during t -- see `momentum_signal`.
    """
    return prices.pct_change(periods=lookback)


def momentum_signal(
    prices: pd.DataFrame | pd.Series, lookback: int = 60
) -> pd.DataFrame | pd.Series:
    """Tradable momentum signal: `momentum_score` lagged by one period.

    The value used for period t is the score computed as of t-1, ensuring the
    decision to be long/short during t relies only on information available
    before t started.
    """
    return momentum_score(prices, lookback=lookback).shift(1)


def momentum_positions(
    prices: pd.DataFrame | pd.Series, lookback: int = 60
) -> pd.DataFrame | pd.Series:
    """Long/short/flat position (+1/-1/0) from the sign of the lagged momentum signal."""
    signal = momentum_signal(prices, lookback=lookback)
    return np.sign(signal)
