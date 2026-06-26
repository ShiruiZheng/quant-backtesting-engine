import numpy as np
import pandas as pd
import pytest

from backtester.core.returns import cumulative_returns, log_returns, simple_returns


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series([100.0, 110.0, 99.0, 99.0, 108.9])


def test_simple_returns_values(prices: pd.Series) -> None:
    result = simple_returns(prices)
    assert result.iloc[0] is np.nan or pd.isna(result.iloc[0])
    np.testing.assert_allclose(result.iloc[1], 0.10)
    np.testing.assert_allclose(result.iloc[2], -0.10)
    np.testing.assert_allclose(result.iloc[3], 0.0)
    np.testing.assert_allclose(result.iloc[4], 0.10)


def test_log_returns_values(prices: pd.Series) -> None:
    result = log_returns(prices)
    assert pd.isna(result.iloc[0])
    np.testing.assert_allclose(result.iloc[1], np.log(1.10))
    np.testing.assert_allclose(result.iloc[2], np.log(99.0 / 110.0))


def test_log_returns_smaller_than_simple_for_positive_moves(prices: pd.Series) -> None:
    simple = simple_returns(prices)
    log = log_returns(prices)
    up_periods = simple > 0
    assert (log[up_periods] < simple[up_periods]).all()


def test_cumulative_returns_compounds(prices: pd.Series) -> None:
    simple = simple_returns(prices)
    cumulative = cumulative_returns(simple)
    expected_final = prices.iloc[-1] / prices.iloc[0] - 1.0
    np.testing.assert_allclose(cumulative.iloc[-1], expected_final, rtol=1e-10)


def test_returns_preserve_index_and_length(prices: pd.Series) -> None:
    result = simple_returns(prices)
    assert len(result) == len(prices)
    assert result.index.equals(prices.index)


def test_returns_work_on_dataframe() -> None:
    df = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 49.5]})
    result = simple_returns(df)
    np.testing.assert_allclose(result["A"].iloc[1:], [0.10, 0.10])
    np.testing.assert_allclose(result["B"].iloc[1:], [-0.10, 0.10])
