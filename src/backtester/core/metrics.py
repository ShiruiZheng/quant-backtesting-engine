"""Performance and risk metrics computed from a periodic returns series.

Every function here takes a Series of *periodic* strategy returns -- the net
per-period returns produced by `backtester.core.engine.run_backtest` -- and
annualizes using `periods_per_year` (252 trading days by default). These are
descriptive summaries of a realized return stream; they carry no lookahead
concerns of their own. Whether the inputs were generated honestly (lagged
signals, out-of-sample windows) is the job of the signal and validation layers.

"Win rate" and "hit rate" are the same quantity here (fraction of up periods);
`hit_rate` is provided as an explicit alias because both names are in common use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaNs so warm-up periods don't poison aggregates."""
    return returns.dropna()


def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole series: prod(1 + r) - 1."""
    r = _clean(returns)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate implied by the periodic returns."""
    r = _clean(returns)
    if r.empty:
        return 0.0
    growth = (1.0 + r).prod()
    years = len(r) / periods_per_year
    if years <= 0 or growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Standard deviation of returns scaled to annual terms by sqrt(periods)."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized mean excess return divided by its volatility.

    `risk_free_rate` is an *annual* rate; it is converted to per-period before
    being subtracted. Returns NaN if there is no variation to divide by.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    # A perfectly flat series has zero true volatility; guard on the raw spread
    # because floating-point error can leave std() at ~1e-18 instead of exactly 0.
    if excess.max() == excess.min():
        return float("nan")
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Like Sharpe, but the denominator only penalizes downside deviation.

    Upside volatility is not "risk" a trader minds, so the denominator is the
    root-mean-square of the *negative* excess returns only.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    downside = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve, as a negative fraction.

    e.g. -0.25 means the strategy lost 25% from a prior high at its worst point.
    """
    r = _clean(returns)
    if r.empty:
        return 0.0
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """CAGR divided by the magnitude of max drawdown -- return per unit of pain."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Value at Risk at the given tail `level`, as a positive loss.

    The (1 - level) confidence VaR: the loss the strategy is not expected to
    exceed on `(1 - level)` of periods, estimated as the empirical `level`
    quantile of returns. A return value of 0.03 means "a 3% periodic loss".
    """
    r = _clean(returns)
    if r.empty:
        return 0.0
    return float(-np.quantile(r, level))


def win_rate(returns: pd.Series) -> float:
    """Fraction of decided periods that were positive: wins / (wins + losses).

    Flat (exactly-zero) periods are excluded from the denominator so a strategy
    that sits in cash a lot is not penalized for it.
    """
    r = _clean(returns)
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    if wins + losses == 0:
        return float("nan")
    return wins / (wins + losses)


def hit_rate(returns: pd.Series) -> float:
    """Alias for `win_rate` -- the two names refer to the same quantity."""
    return win_rate(returns)


def summarize(
    returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    var_level: float = 0.05,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """Bundle the standard metric set into a single dict (handy for JSON/reports)."""
    summary: dict[str, float] = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        f"var_{int(var_level * 100)}pct": value_at_risk(returns, var_level),
        "win_rate": win_rate(returns),
    }
    if extra:
        summary.update(extra)
    return summary
