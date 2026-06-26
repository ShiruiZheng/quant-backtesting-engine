import numpy as np
import pandas as pd

from backtester.signals.pairs import (
    engle_granger,
    pairs_positions,
    spread_zscore,
)


def _cointegrated_pair(n: int = 400, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """x is a random walk (non-stationary); y = 3 + 2*x + stationary noise."""
    rng = np.random.default_rng(seed)
    x = pd.Series(50.0 + np.cumsum(rng.normal(0.0, 1.0, size=n)), name="X")
    y = pd.Series(3.0 + 2.0 * x.to_numpy() + rng.normal(0.0, 0.5, size=n), name="Y")
    return y, x


def test_engle_granger_recovers_hedge_ratio() -> None:
    y, x = _cointegrated_pair()
    result = engle_granger(y, x)
    np.testing.assert_allclose(result.hedge_ratio, 2.0, atol=0.1)
    np.testing.assert_allclose(result.intercept, 3.0, atol=1.0)


def test_engle_granger_detects_cointegration() -> None:
    y, x = _cointegrated_pair()
    result = engle_granger(y, x)
    assert result.is_cointegrated(alpha=0.05)
    assert result.adf_pvalue < 0.05


def test_independent_random_walks_are_not_cointegrated() -> None:
    rng = np.random.default_rng(123)
    n = 400
    x = pd.Series(np.cumsum(rng.normal(0.0, 1.0, size=n)), name="X")
    y = pd.Series(np.cumsum(rng.normal(0.0, 1.0, size=n)), name="Y")
    result = engle_granger(y, x)
    assert not result.is_cointegrated(alpha=0.05)


def test_engle_granger_raises_without_overlap() -> None:
    y = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    x = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    try:
        engle_granger(y, x)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected ValueError for non-overlapping series")


def test_spread_zscore_standardizes_to_recent_window() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0, 10.0])
    z = spread_zscore(spread, lookback=4)
    # Last point is far above its trailing window mean -> strongly positive z.
    assert z.iloc[-1] > 1.0
    assert z.iloc[:3].isna().all()


def test_pairs_positions_columns_and_shift() -> None:
    y, x = _cointegrated_pair()
    positions = pairs_positions(y, x, lookback=20, hedge_ratio=2.0)
    assert list(positions.columns) == ["Y", "X"]
    # First row is NaN because of the mandatory one-period shift.
    assert positions.iloc[0].isna().all()


def test_pairs_positions_hedges_x_against_y() -> None:
    y, x = _cointegrated_pair()
    positions = pairs_positions(y, x, lookback=20, hedge_ratio=2.0).dropna()
    # The x leg is always -hedge_ratio times the y leg (short the hedge).
    np.testing.assert_allclose(positions["X"].to_numpy(), -2.0 * positions["Y"].to_numpy())


def test_pairs_positions_no_lookahead() -> None:
    """Perturbing y at time t must not change any position at or before t.

    A fixed hedge_ratio is passed so the (in-sample) OLS refit doesn't repaint
    history; this isolates the rolling z-score + shift logic.
    """
    y, x = _cointegrated_pair()
    t = len(y) // 2

    base = pairs_positions(y, x, lookback=20, hedge_ratio=2.0)
    perturbed_y = y.copy()
    perturbed_y.iloc[t] *= 1.5
    perturbed = pairs_positions(perturbed_y, x, lookback=20, hedge_ratio=2.0)

    pd.testing.assert_frame_equal(base.iloc[: t + 1], perturbed.iloc[: t + 1])
