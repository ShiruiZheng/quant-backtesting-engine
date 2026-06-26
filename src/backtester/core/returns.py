"""Vectorized return calculations from a price series.

These are descriptive period-over-period returns computed directly from prices.
They describe what *did* happen over period t-1 -> t. Whether a return is safe
to use as a trading input depends on when it becomes known relative to the
position being held -- see `backtester.signals` for the shift applied there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple period return: (p_t / p_{t-1}) - 1. First row is NaN."""
    return prices.pct_change()


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log period return: ln(p_t / p_{t-1}). First row is NaN."""
    return np.log(prices / prices.shift(1))


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compounded growth of 1 unit of capital from a series of simple period returns."""
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0
