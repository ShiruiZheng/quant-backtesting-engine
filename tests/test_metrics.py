import numpy as np
import pandas as pd

from backtester.core.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
    value_at_risk,
    win_rate,
)


def test_total_return_compounds() -> None:
    r = pd.Series([0.10, -0.10])
    # (1.1 * 0.9) - 1 = -0.01
    np.testing.assert_allclose(total_return(r), -0.01)


def test_total_return_ignores_nan_warmup() -> None:
    r = pd.Series([np.nan, np.nan, 0.10, 0.10])
    np.testing.assert_allclose(total_return(r), 1.1 * 1.1 - 1.0)


def test_cagr_zero_for_flat_returns() -> None:
    r = pd.Series([0.0] * 252)
    np.testing.assert_allclose(cagr(r), 0.0)


def test_cagr_positive_for_growth() -> None:
    r = pd.Series([0.001] * 252)
    assert cagr(r) > 0.0


def test_annualized_volatility_matches_manual() -> None:
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    expected = float(np.std(r.to_numpy(), ddof=1) * np.sqrt(252))
    np.testing.assert_allclose(annualized_volatility(r), expected)


def test_sharpe_is_nan_when_no_variation() -> None:
    r = pd.Series([0.01] * 10)
    assert np.isnan(sharpe_ratio(r))


def test_sharpe_positive_for_positive_mean() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.005, size=500))
    assert sharpe_ratio(r) > 0.0


def test_sortino_only_penalizes_downside() -> None:
    # Same series but with a big *upside* spike: Sortino should not fall (upside
    # is not "risk"), whereas plain volatility would rise.
    base = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    spiked = base.copy()
    spiked.iloc[0] = 0.50  # large positive move only
    assert sortino_ratio(spiked) >= sortino_ratio(base)


def test_max_drawdown_known_path() -> None:
    # equity: 1.0 -> 0.5 -> 1.0 ; worst drawdown is -50%.
    r = pd.Series([0.0, -0.5, 1.0])
    np.testing.assert_allclose(max_drawdown(r), -0.5)


def test_max_drawdown_non_positive() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.02, size=300))
    assert max_drawdown(r) <= 0.0


def test_calmar_is_cagr_over_drawdown() -> None:
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 20)
    expected = cagr(r) / abs(max_drawdown(r))
    np.testing.assert_allclose(calmar_ratio(r), expected)


def test_value_at_risk_at_zero_level_is_worst_loss() -> None:
    r = pd.Series([-0.2, -0.1, 0.0, 0.1, 0.2])
    # quantile at level 0.0 is the minimum (-0.2); VaR is reported as a positive loss.
    np.testing.assert_allclose(value_at_risk(r, level=0.0), 0.2)


def test_win_rate_excludes_flat_periods() -> None:
    r = pd.Series([0.1, -0.1, 0.0, 0.2, -0.3])  # 2 wins, 2 losses, 1 flat
    np.testing.assert_allclose(win_rate(r), 0.5)


def test_hit_rate_is_alias_for_win_rate() -> None:
    r = pd.Series([0.1, -0.1, 0.2])
    assert hit_rate(r) == win_rate(r)


def test_summarize_has_expected_keys() -> None:
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 20)
    summary = summarize(r)
    expected = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "var_5pct",
        "win_rate",
    }
    assert expected.issubset(summary.keys())


def test_summarize_accepts_extra_metrics() -> None:
    r = pd.Series([0.01, -0.01, 0.02])
    summary = summarize(r, extra={"avg_turnover": 0.3})
    assert summary["avg_turnover"] == 0.3


def test_metrics_handle_empty_series() -> None:
    empty = pd.Series(dtype=float)
    assert total_return(empty) == 0.0
    assert max_drawdown(empty) == 0.0
    assert annualized_volatility(empty) == 0.0
