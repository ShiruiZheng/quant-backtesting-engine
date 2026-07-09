import numpy as np
import pandas as pd
import pytest

from backtester.signals.momentum import momentum_positions, momentum_score, momentum_signal


@pytest.fixture
def prices() -> pd.Series:
    rng = np.random.default_rng(seed=42)
    n = 50
    walk = np.cumsum(rng.normal(loc=0.001, scale=0.01, size=n))
    return pd.Series(100.0 * np.exp(walk), name="PRICE")


def test_signal_is_score_shifted_by_exactly_one_period(prices: pd.Series) -> None:
    lookback = 5
    score = momentum_score(prices, lookback=lookback)
    signal = momentum_signal(prices, lookback=lookback)
    # signal(t) must equal score(t-1): the value used to trade at t was computed at t-1.
    pd.testing.assert_series_equal(
        signal.iloc[1:].reset_index(drop=True), score.iloc[:-1].reset_index(drop=True)
    )


def test_signal_does_not_use_same_day_price(prices: pd.Series) -> None:
    """Changing today's price must not change today's signal (no same-bar lookahead)."""
    lookback = 5
    t = 30
    perturbed = prices.copy()
    perturbed.iloc[t] *= 1.5  # shock the price at t only

    signal_original = momentum_signal(prices, lookback=lookback)
    signal_perturbed = momentum_signal(perturbed, lookback=lookback)

    pd.testing.assert_series_equal(signal_original.iloc[: t + 1], signal_perturbed.iloc[: t + 1])


def test_signal_does_not_use_future_prices(prices: pd.Series) -> None:
    """Changing prices strictly after t must not change the signal at or before t."""
    lookback = 5
    cutoff = 25
    perturbed = prices.copy()
    perturbed.iloc[cutoff + 1 :] = perturbed.iloc[cutoff + 1 :] * 3.0 + 1000.0

    signal_original = momentum_signal(prices, lookback=lookback)
    signal_perturbed = momentum_signal(perturbed, lookback=lookback)

    pd.testing.assert_series_equal(
        signal_original.iloc[: cutoff + 1], signal_perturbed.iloc[: cutoff + 1]
    )


def test_first_lookback_plus_one_rows_are_nan(prices: pd.Series) -> None:
    lookback = 5
    signal = momentum_signal(prices, lookback=lookback)
    assert signal.iloc[: lookback + 1].isna().all()
    assert signal.iloc[lookback + 1 :].notna().all()


def test_positions_are_sign_of_signal(prices: pd.Series) -> None:
    lookback = 5
    signal = momentum_signal(prices, lookback=lookback)
    positions = momentum_positions(prices, lookback=lookback)
    valid = signal.notna()
    assert set(positions[valid].unique()).issubset({-1.0, 0.0, 1.0})
    np.testing.assert_array_equal(positions[valid].to_numpy(), np.sign(signal[valid].to_numpy()))


def test_dataframe_input_preserves_columns() -> None:
    df = pd.DataFrame(
        {
            "BHP.AX": [10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.3],
            "CBA.AX": [100.0, 99.0, 98.0, 97.5, 99.0, 101.0, 102.0],
        }
    )
    signal = momentum_signal(df, lookback=2)
    assert list(signal.columns) == ["BHP.AX", "CBA.AX"]
    assert signal.iloc[:3].isna().all().all()
