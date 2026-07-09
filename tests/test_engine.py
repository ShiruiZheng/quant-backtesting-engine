import numpy as np
import pandas as pd
import pytest

from backtester.core.costs import CostModel
from backtester.core.engine import run_backtest


@pytest.fixture
def up_prices() -> pd.DataFrame:
    # A single asset compounding at exactly +10% per period.
    return pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]})


def test_long_position_compounds_returns_without_costs(up_prices: pd.DataFrame) -> None:
    positions = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]})
    result = run_backtest(up_prices, positions, initial_capital=10_000.0)
    # 10% per period for 3 realized periods: 10000 * 1.1**3 = 13310.
    np.testing.assert_allclose(result.final_equity, 13_310.0)


def test_position_at_t_earns_return_of_t_not_t_plus_one() -> None:
    # The engine must NOT shift positions itself (signals already do).
    prices = pd.DataFrame({"A": [100.0, 110.0]})
    positions = pd.DataFrame({"A": [0.0, 1.0]})
    result = run_backtest(prices, positions)
    # Held flat over period 0, fully long over period 1 (which returns +10%).
    np.testing.assert_allclose(result.gross_returns.to_numpy(), [0.0, 0.10])


def test_turnover_counts_initial_entry_and_changes() -> None:
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]})
    positions = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0]})
    result = run_backtest(prices, positions)
    # changes vs prior (row 0 is entry from flat): [0, +1, 0, -1] -> abs turnover.
    np.testing.assert_allclose(result.turnover.to_numpy(), [0.0, 1.0, 0.0, 1.0])


def test_costs_reduce_net_return_below_gross() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0]})
    positions = pd.DataFrame({"A": [1.0, 1.0]})
    model = CostModel(proportional_bps=10.0, slippage_bps=5.0)  # 15 bps on turnover
    result = run_backtest(prices, positions, cost_model=model)
    # Period 0 establishes the position: turnover 1.0 -> cost 15 bps = 0.0015.
    np.testing.assert_allclose(result.costs.iloc[0], 0.0015)
    assert (result.returns <= result.gross_returns + 1e-12).all()


def test_zero_cost_model_leaves_returns_gross(up_prices: pd.DataFrame) -> None:
    positions = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]})
    result = run_backtest(up_prices, positions)
    np.testing.assert_allclose(result.returns.to_numpy(), result.gross_returns.to_numpy())


def test_series_input_is_supported() -> None:
    prices = pd.Series([100.0, 110.0, 121.0], name="A")
    positions = pd.Series([1.0, 1.0, 1.0], name="A")
    result = run_backtest(prices, positions)
    np.testing.assert_allclose(result.final_equity, 10_000.0 * 1.1**2)


def test_multi_asset_portfolio_sums_weighted_returns() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 90.0]})
    positions = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    result = run_backtest(prices, positions)
    # period 1: 0.5*(+0.10) + 0.5*(-0.10) = 0.0
    np.testing.assert_allclose(result.gross_returns.iloc[1], 0.0, atol=1e-12)


def test_misaligned_positions_treated_as_flat() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    # Positions missing the last row -> reindexed and filled flat (no crash, no trade).
    positions = pd.DataFrame({"A": [1.0, 1.0]}, index=[0, 1])
    result = run_backtest(prices, positions)
    assert len(result.equity_curve) == 3
