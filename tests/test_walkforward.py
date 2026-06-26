import numpy as np
import pandas as pd
import pytest

from backtester.validation.walkforward import walk_forward


@pytest.fixture
def prices() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=120)
    rng = np.random.default_rng(0)
    a = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=120)))
    b = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=120)))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _always_long(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # Equal-weight, fully invested over the whole test window.
    return pd.DataFrame(0.5, index=test.index, columns=test.columns)


def test_walk_forward_window_count_and_stitch_length(prices: pd.DataFrame) -> None:
    result = walk_forward(prices, _always_long, train_size=40, test_size=20, step=20)
    # starts at 0, 20, 40, 60 (each needs start+60 <= 120) -> 4 windows.
    assert result.n_windows == 4
    assert len(result.oos_returns) == 4 * 20


def test_walk_forward_oos_index_is_ordered_and_unique(prices: pd.DataFrame) -> None:
    result = walk_forward(prices, _always_long, train_size=40, test_size=20, step=20)
    idx = result.oos_returns.index
    assert idx.is_monotonic_increasing
    assert idx.is_unique


def test_walk_forward_metrics_present(prices: pd.DataFrame) -> None:
    result = walk_forward(prices, _always_long, train_size=40, test_size=20, step=20)
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics
    # Per-window records carry their date ranges for inspection.
    assert "test_start" in result.window_metrics[0]


def test_walk_forward_raises_when_too_little_data(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward(prices, _always_long, train_size=100, test_size=50)


def test_walk_forward_equity_curve_tracks_returns(prices: pd.DataFrame) -> None:
    result = walk_forward(
        prices, _always_long, train_size=40, test_size=20, step=20, initial_capital=1_000.0
    )
    expected_final = 1_000.0 * float((1.0 + result.oos_returns).prod())
    np.testing.assert_allclose(result.equity_curve.iloc[-1], expected_final)
