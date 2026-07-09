import numpy as np
import pandas as pd
import pytest

from backtester.core.costs import CostModel


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series([10.0, 10.0, 10.0, 10.0])


def test_no_trade_means_no_cost(prices: pd.Series) -> None:
    model = CostModel(fixed_fee=5.0, proportional_bps=10, slippage_bps=5, market_impact_coef=0.01)
    position_changes = pd.Series([0.0, 0.0, 0.0, 0.0])
    costs = model.trade_costs(position_changes, prices)
    np.testing.assert_array_equal(costs.to_numpy(), 0.0)


def test_fixed_fee_charged_once_per_trade(prices: pd.Series) -> None:
    model = CostModel(fixed_fee=5.0)
    position_changes = pd.Series([1.0, 0.0, -2.0, 0.0])
    costs = model.trade_costs(position_changes, prices)
    np.testing.assert_array_equal(costs.to_numpy(), [5.0, 0.0, 5.0, 0.0])


def test_fixed_fee_independent_of_trade_size(prices: pd.Series) -> None:
    model = CostModel(fixed_fee=5.0)
    small_trade = model.trade_costs(pd.Series([1.0]), pd.Series([10.0]))
    large_trade = model.trade_costs(pd.Series([100.0]), pd.Series([10.0]))
    np.testing.assert_allclose(small_trade.to_numpy(), large_trade.to_numpy())


def test_proportional_cost_scales_linearly_with_notional(prices: pd.Series) -> None:
    model = CostModel(proportional_bps=10.0)  # 0.10%
    position_changes = pd.Series([1.0, 0.0, 2.0, 0.0])
    costs = model.trade_costs(position_changes, prices)
    # notional = traded_size * price = [10, 0, 20, 0]; cost = notional * 10bps
    np.testing.assert_allclose(costs.to_numpy(), [0.01, 0.0, 0.02, 0.0])


def test_slippage_cost_scales_linearly_with_notional(prices: pd.Series) -> None:
    model = CostModel(slippage_bps=20.0)  # 0.20%
    position_changes = pd.Series([1.0, 0.0, 2.0, 0.0])
    costs = model.trade_costs(position_changes, prices)
    np.testing.assert_allclose(costs.to_numpy(), [0.02, 0.0, 0.04, 0.0])


def test_market_impact_scales_quadratically_with_trade_size(prices: pd.Series) -> None:
    model = CostModel(market_impact_coef=0.001)
    one_unit = model.trade_costs(pd.Series([1.0]), pd.Series([10.0])).iloc[0]
    two_units = model.trade_costs(pd.Series([2.0]), pd.Series([10.0])).iloc[0]
    # impact = coef * size^2 * price -> doubling size should quadruple the cost.
    np.testing.assert_allclose(two_units, one_unit * 4.0)


def test_costs_compose_additively(prices: pd.Series) -> None:
    model = CostModel(
        fixed_fee=1.0, proportional_bps=10.0, slippage_bps=5.0, market_impact_coef=0.01
    )
    position_changes = pd.Series([3.0])
    price = pd.Series([10.0])

    fixed_only = CostModel(fixed_fee=1.0).trade_costs(position_changes, price)
    proportional_only = CostModel(proportional_bps=10.0).trade_costs(position_changes, price)
    slippage_only = CostModel(slippage_bps=5.0).trade_costs(position_changes, price)
    impact_only = CostModel(market_impact_coef=0.01).trade_costs(position_changes, price)

    combined = model.trade_costs(position_changes, price)
    expected = fixed_only + proportional_only + slippage_only + impact_only
    np.testing.assert_allclose(combined.to_numpy(), expected.to_numpy())


def test_costs_work_on_dataframe() -> None:
    model = CostModel(fixed_fee=1.0, proportional_bps=10.0)
    position_changes = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 2.0]})
    prices = pd.DataFrame({"A": [10.0, 10.0], "B": [5.0, 5.0]})
    costs = model.trade_costs(position_changes, prices)
    assert list(costs.columns) == ["A", "B"]
    np.testing.assert_allclose(costs["A"].to_numpy(), [1.0 + 10.0 * 10.0 / 1e4, 0.0])
    np.testing.assert_allclose(costs["B"].to_numpy(), [0.0, 1.0 + 10.0 * 10.0 / 1e4])


def test_zero_cost_model_is_free(prices: pd.Series) -> None:
    model = CostModel()
    position_changes = pd.Series([5.0, -5.0, 3.0, 0.0])
    costs = model.trade_costs(position_changes, prices)
    np.testing.assert_array_equal(costs.to_numpy(), 0.0)


def test_trade_cost_fraction_uses_bps_on_weight_change() -> None:
    model = CostModel(proportional_bps=10.0, slippage_bps=5.0)  # 15 bps total
    weight_changes = pd.Series([0.5, -0.5, 0.0])
    costs = model.trade_cost_fraction(weight_changes)
    # |delta_weight| * 15bps = |delta_weight| * 0.0015
    np.testing.assert_allclose(costs.to_numpy(), [0.00075, 0.00075, 0.0])


def test_trade_cost_fraction_ignores_fixed_and_impact() -> None:
    # fixed_fee and market_impact are currency/share-space; they don't enter weight space.
    bps_only = CostModel(proportional_bps=10.0, slippage_bps=5.0)
    with_extras = CostModel(
        proportional_bps=10.0, slippage_bps=5.0, fixed_fee=100.0, market_impact_coef=1.0
    )
    weight_changes = pd.Series([0.3, -0.2])
    np.testing.assert_allclose(
        bps_only.trade_cost_fraction(weight_changes).to_numpy(),
        with_extras.trade_cost_fraction(weight_changes).to_numpy(),
    )


def test_trade_cost_fraction_works_on_dataframe() -> None:
    model = CostModel(proportional_bps=20.0)  # 20 bps
    weight_changes = pd.DataFrame({"A": [0.5, 0.0], "B": [0.0, -1.0]})
    costs = model.trade_cost_fraction(weight_changes)
    assert list(costs.columns) == ["A", "B"]
    np.testing.assert_allclose(costs["A"].to_numpy(), [0.5 * 0.002, 0.0])
    np.testing.assert_allclose(costs["B"].to_numpy(), [0.0, 1.0 * 0.002])
