# Concepts

Running notes on the ideas behind this project, in plain language. Updated as
new pieces get built — see [[devlog.md]] for the day-by-day log of when each
concept showed up.

## Backtest

Simulating a trading strategy against historical price data to see what
positions and returns it *would have produced*. It is a research tool, not a
guarantee — a strategy that worked in the past may not work going forward,
and a poorly built backtest (e.g. with lookahead bias) can look profitable
purely because it's cheating.

## Ticker

The exchange code for a tradable instrument, e.g. `BHP` for BHP Group. Codes
are exchange-specific: the same company can have different tickers (or none)
on different exchanges. yfinance expects ASX-listed tickers with a `.AX`
suffix (`BHP.AX`), which `backtester.data.loader.to_asx_ticker` adds
automatically.

## Lookahead bias

Using information in a trading decision that would not actually have been
known at the time the decision was made. It's the single most common way a
backtest silently overstates performance, because the model gets to "see the
future" in small, easy-to-miss ways: computing a signal from a price and then
using it to size a position *during the same period* that price was observed.

The fix used throughout this codebase: a signal computed from data known by
the close of period `t-1` is the only thing allowed to determine the
position held *during* period `t`. Concretely, `momentum_signal` always
calls `.shift(1)` on the raw `momentum_score` before returning it — see
[[decisions.md]] for why that shift lives inside the signal function instead
of being left to the caller, and the no-lookahead tests in
`tests/test_momentum.py` for how this is actually verified (perturbing a
price at or after time `t` and checking the signal up to `t` is unchanged).

## Simple vs. log returns

- Simple return: `p_t / p_{t-1} - 1`. Directly interpretable as "% change",
  and the right form for things like cost calculations that work in
  notional currency.
- Log return: `ln(p_t / p_{t-1})`. Additive across time (the sum of log
  returns over a period equals the log return of the whole period), which
  makes them convenient for statistics, but they're an approximation of the
  simple return that understates gains and overstates losses slightly for
  larger moves.

## Transaction cost components

- **Fixed fee**: a flat cost charged whenever a trade happens, regardless of
  size (e.g. brokerage commission).
- **Proportional fee (bps)**: a cost proportional to traded notional, quoted
  in basis points (1 bps = 0.01%).
- **Slippage**: modeled the same way as a proportional fee here — the
  difference between the price you wanted and the price you got, often from
  bid/ask spread.
- **Market impact**: the idea that your own trade moves the price against
  you, and that this effect grows faster than linearly with trade size. This
  project uses a simplified quadratic-in-size approximation
  (`coef * size^2 * price`) rather than a calibrated model — good enough to
  demonstrate the concept, not a real microstructure model.

## Forward-fill vs. interpolation for missing data

Forward-filling a missing price copies the most recent *past* value forward
in time. Interpolation (e.g. linear) can use a *future* value to estimate a
past gap — which is itself a (subtle) form of lookahead bias. The loader
forward-fills for this reason.

## Equity curve / PnL simulation

The thing a "backtest engine" actually produces: starting from some capital,
apply each period's strategy return and compound forward into an account value
over time. In `backtester.core.engine`, the per-period strategy return is
`sum_i weight_i(t) * asset_return_i(t)` minus transaction costs, and the equity
curve is `initial_capital * cumprod(1 + net_return)`. The engine assumes the
positions handed to it are *already* lagged (lookahead-safe) and does not shift
them again — see [[decisions.md]].

## Weights vs. share counts

This engine treats positions as portfolio **weights** (fraction of capital per
asset), not share counts. That keeps the whole simulation unit-consistent: a
trade of `|Δweight|` moves that fraction of capital, returns combine as a
weighted average, and the equity curve is dimensionless growth scaled by
starting capital. The `CostModel.trade_cost_fraction` helper charges the linear
(bps) costs directly on `|Δweight|`; the older currency-based `trade_costs`
(fixed fee, market impact) stays for share/notional-space work.

## Turnover

The total amount of position change in a period, `sum_i |weight_i(t) -
weight_i(t-1)|`. It is what transaction costs are charged on, and a quick proxy
for how much a strategy trades (and therefore how cost-sensitive it is).
Establishing the initial position from flat counts as turnover.

## Sharpe / Sortino / Calmar

Risk-adjusted return ratios, all in `backtester.core.metrics`:

- **Sharpe**: annualized mean excess return ÷ volatility of returns. Reward per
  unit of total risk.
- **Sortino**: like Sharpe but the denominator only counts *downside*
  deviation — upside swings aren't "risk" a trader wants to penalize.
- **Calmar**: annualized return (CAGR) ÷ |max drawdown|. Reward per unit of
  worst-case pain.

## Max drawdown

The largest peak-to-trough decline of the equity curve, reported as a negative
fraction (e.g. `-0.25` = lost 25% from a prior high at the worst point). A blunt
but very intuitive measure of "how bad did it get".

## VaR (Value at Risk)

A loss threshold not expected to be exceeded with some confidence over a period.
This project uses *historical* VaR: the empirical `level` quantile of the return
distribution (e.g. the 5th percentile), reported as a positive loss number.

## Win rate / hit rate

The fraction of decided periods that were positive: `wins / (wins + losses)`,
excluding exactly-flat periods. The two names mean the same thing here.

## Cointegration & Engle-Granger

Two price series are **cointegrated** if some linear combination of them is
stationary (mean-reverting) even though each series individually wanders
(is non-stationary). Engle-Granger tests this in two steps: regress one on the
other to get a **hedge ratio**, then run an Augmented Dickey-Fuller (ADF) test
on the residual **spread**; a low ADF p-value is evidence of mean reversion.
This is the basis of the pairs/stat-arb signal in `backtester.signals.pairs`.

## Z-score (of a spread)

Standardizing a value by its recent history: `(x - rolling_mean) / rolling_std`.
The pairs signal trades the spread's z-score — fade it when it's far from zero
(unusually wide) and flatten as it reverts. A trailing (causal) window is used,
and the resulting position is still shifted one period before trading.

## Walk-forward validation

Repeatedly fit/choose parameters on a trailing **train** window and evaluate on
the next out-of-sample **test** window, rolling forward through time. Stitching
the test windows gives one continuous out-of-sample track record — far closer to
"how it would have behaved if deployed" than a single split. Implemented in
`backtester.validation.walk_forward`; the gap between in-sample and
walk-forward metrics is the clearest signal of overfitting this project has.

## Still ahead (not yet implemented)

Honest list of what a research-grade engine has that this one still doesn't:

- **Caching / storage layer**: persist fetched prices (Parquet/DuckDB) instead
  of re-hitting yfinance every run.
- **Survivorship bias**: yfinance only serves still-listed names, so backtests
  here quietly overstate returns by ignoring delisted companies.
- **Point-in-time data**: using only what was actually known on each date (no
  later-revised fundamentals leaking back).
- **Calibrated market impact / portfolio optimization / multi-factor models**:
  the cost model and position sizing are deliberately simple. See the answer in
  [[devlog.md]] (2026-06-25 session 2) for the fuller roadmap.
