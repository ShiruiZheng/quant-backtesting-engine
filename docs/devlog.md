# Devlog

## 2026-06-25

### Built

- Project skeleton: `src/backtester/` package (`data/`, `core/`, `signals/`), `tests/`, `docs/`.
- `pyproject.toml` (setuptools, src-layout, pytest + ruff config, `dev` extra).
- `core/returns.py`: vectorized `simple_returns`, `log_returns`, `cumulative_returns`.
- `signals/momentum.py`: `momentum_score` (trailing return) -> `momentum_signal`
  (shifted by 1 period) -> `momentum_positions` (sign of the shifted signal).
- `core/costs.py`: `CostModel` dataclass — fixed fee, proportional bps, slippage bps,
  and a simplified quadratic-in-size market impact term — applied vectorized to a
  position-changes series.
- `data/loader.py`: yfinance-backed loader, `.AX` suffix handling, per-ticker
  fetch so one bad ticker doesn't fail the whole batch, forward-fill for gaps.
- pytest suite (29 tests passing): returns correctness, no-lookahead guarantees
  for the momentum signal, cost-model arithmetic, and a mocked (no-network) loader.
- `.github/workflows/ci.yml`: runs on push to `main` and on PRs —
  `uv sync --extra dev --locked` (fails if `uv.lock` is stale), then
  `ruff check .`, then `pytest -q`.
- `scripts/run_demo.py`: manual runnable example wiring `load_prices` ->
  `momentum_positions` -> `CostModel.trade_costs` together against live
  yfinance data. Confirmed working end-to-end against real ASX history
  (BHP.AX, CBA.AX). Documented in the new README quickstart section.

### Learned

- A backtest simulates a strategy against historical data to see what its
  positions and P&L *would have been*; it says nothing about future performance.
- A ticker is the exchange code for a tradable instrument; ASX tickers need a
  `.AX` suffix for yfinance (e.g. `BHP` -> `BHP.AX`).
- Lookahead bias: using information that would not have been available at the
  time a trade decision was made. The standard guard is to shift any signal
  derived from price/return data by one period before using it to size a
  position for that period. See [[concepts.md]] for the full explanation and
  [[decisions.md]] for why this is enforced inside `momentum_signal` itself
  rather than left to the caller.
- Forward-filling missing price data only ever copies a *past* value forward,
  so it does not introduce lookahead bias — unlike interpolation, which can
  use a future value to fill a past gap.

### Problems

- yfinance ASX tickers need the `.AX` suffix; passing a bare code like `BHP`
  either fails or silently resolves to the wrong (US) listing.
- Thinly-traded ASX tickers can have missing trading days; `load_prices`
  forward-fills these and drops only the dates where *every* requested ticker
  is missing.
- `uv` and Python 3.11 were not actually installed on this machine despite the
  original plan, so started on pip + venv with Python 3.12.12 instead.
- After installing `uv`, switched the project over to it the same day:
  `uv python install 3.11` + `uv python pin 3.11` + `uv sync --extra dev`.
  `uv.lock` is now committed. See [[decisions.md]] for the full reasoning.

### Next

- Engle-Granger cointegration pairs/stat-arb signal.
- Risk metrics module: Sharpe, Sortino, Calmar, max drawdown, VaR, win rate.
- Walk-forward validation harness.
- Vectorized backtest "engine" that ties signals + costs + prices into an
  equity curve (`run_demo.py` shows the pieces composing, but there's still
  no PnL/equity-curve simulation behind it).
- FastAPI endpoint, Dockerfile, docker-compose.

## 2026-06-26 (session 2 — engine, metrics, pairs, walk-forward, API)

### Built

- Switched the demo to **US equities** by default (`AAPL MSFT GOOGL AMZN NVDA`);
  `load_prices(..., asx=False)` already existed, so this was a `run_demo.py`
  change plus an `--asx` flag to keep Australian listings available.
- `core/metrics.py`: Sharpe, Sortino, Calmar, max drawdown, CAGR, total return,
  annualized volatility, historical VaR, win/hit rate, and a `summarize()`
  bundle. All operate on a periodic returns Series.
- `core/engine.py`: `run_backtest` — the missing PnL/equity-curve simulator.
  Positions (as weights) × next-period returns − costs → equity curve, returned
  as a `BacktestResult` with a `.summary()`.
- `core/costs.py`: added `trade_cost_fraction` (weight-space bps cost) so the
  cost model composes cleanly with the weight-based engine; left `trade_costs`
  (currency) untouched.
- `signals/pairs.py`: Engle-Granger cointegration (`engle_granger`), spread
  z-score, and a lagged `pairs_positions` stat-arb signal.
- `validation/walkforward.py`: signal-agnostic rolling train/test harness
  (`walk_forward`) producing stitched out-of-sample returns + per-window metrics.
- `api/app.py`: FastAPI service — `GET /health`, `POST /backtest`.
- `Dockerfile` + `docker-compose.yml` (+ `.dockerignore`): containerized API on
  uv, installed against the committed lockfile.
- Tests for every new module (metrics, engine, pairs, walk-forward, API,
  weight-space costs); suite now at 73 passing, still fully no-network (the API
  test monkeypatches the loader).

### Learned

- A backtest "engine" is really just the **equity-curve simulation**: everything
  before it (loader, signal, cost model) only produces inputs; the engine is
  what compounds positions × returns − costs into PnL.
- Doing the engine in **weights** (fraction of capital) rather than share counts
  keeps units consistent and makes costs = `|Δweight| × bps`.
- **Walk-forward is the honest test.** On the demo data the in-sample momentum
  Sharpe (~1.3) collapses out-of-sample (~0.06) — the in-sample number was
  mostly overfit. This is the concrete version of "real quants use deeper
  statistical controls".
- **Cointegration ≠ correlation**: Engle-Granger looks for a *stationary spread*
  (a long-run equilibrium), which is what a pairs trade actually needs.

### Problems

- `sharpe_ratio` on a perfectly flat returns series exploded instead of
  returning NaN: identical floats still leave `std()` at ~1e-18 because of
  `mean = sum/n` rounding, so the `sd == 0` guard missed it. Fixed by guarding
  on the raw spread (`max == min`), which is exact. See `tests/test_metrics.py`.
- Float dust in a multi-asset gross-return test (`5.5e-17` vs `0.0`) — added an
  `atol`. Both were test/edge issues, not engine bugs.

### Next

- Caching/storage layer (Parquet/DuckDB) so runs don't re-fetch from yfinance.
- Address survivorship bias / point-in-time data for honest long backtests.
- Richer position sizing (volatility targeting) and a calibrated cost model.

## 2026-06-26 (session 3 — CI/CD & type checking)

### Built

- Full CI/CD pipeline. CI (`.github/workflows/ci.yml`) now also runs `mypy`
  (added a `src/backtester/py.typed` marker so the package is actually
  type-checked). CD (`.github/workflows/cd.yml`) builds the Docker image,
  smoke-tests that the container answers `/health`, then publishes it to GHCR
  (GitHub Container Registry) on push to `main`, on `v*` tags, or via manual
  dispatch from the Actions tab.
- Verified the whole chain locally before committing: `docker build` succeeds,
  the running container returns `{"status":"ok"}` from `/health`, and `mypy` is
  clean across all 15 source files (76 tests still green, ruff clean).
- Committed `docs/ideas.md` (research notes: Monte Carlo, parameter sensitivity,
  walk-forward, overfitting controls, log returns, public-strategy edge sources).

### Learned

- "Full CI/CD" = CI (lint + type-check + test on every PR) **plus** CD (build a
  versioned container artifact, prove it boots, publish it). Publishing to GHCR
  needs no external secrets — the built-in `GITHUB_TOKEN` has `packages: write`.
- A type checker only helps if it actually runs: a library needs a `py.typed`
  marker (PEP 561) or tools skip it ("missing py.typed marker").
- A CD **smoke test** (boot the image, curl `/health`) catches what unit tests
  can't — a broken Dockerfile or start command.

### Problems

- `mypy` initially type-checked *nothing* ("missing py.typed marker"); fixed by
  adding the marker. No live deployment *target* (Fly/Render/AWS) is wired up:
  CD publishes a runnable image, but pointing a host at it is a manual step.

### Next

- Point a host (Fly.io / Render) at the GHCR image for a live URL.
- Parameter-sensitivity sweep + Monte Carlo resampling — see [[ideas.md]].

## 2026-06-26 (session 4 — PriceSource interface + DuckDB cache)

### Built

- `data/sources.py`: a `PriceSource` Protocol with two implementations —
  `YFinanceSource` (live) and `CachedPriceSource` (DuckDB-backed). This removes
  the one real coupling smell (everything was wired straight to yfinance).
- Refactored `data/loader.py`: `load_prices` now delegates fetching to a
  `PriceSource` (new optional `source=` arg, defaults to `YFinanceSource`) and
  keeps only ticker-name resolution + the single forward-fill. Same public
  signature, nothing downstream changed.
- Wired the cache in: the demo caches at `.cache/prices.duckdb` (`--no-cache`
  to disable); the API uses a lazily-built cached source shared across requests
  (`$BACKTESTER_CACHE_PATH`).
- Added `duckdb` dependency (via `uv add`, so pyproject + uv.lock stay in sync).
- Tests: new `tests/test_cache.py` (serves-from-cache, persistence across
  instances, sub-range hit, range-extension fetch, unknown-ticker raise);
  updated `tests/test_loader.py` to patch yfinance in its new home
  (`sources.yf`). 83 tests pass; ruff + mypy clean.

### Learned

- **Dependency inversion**: depend on a small interface (`PriceSource`), not on
  yfinance/DuckDB. The cache then drops in as another implementation instead of
  forcing a rewrite — the payoff of the abstraction.
- An interface earns its keep at **two** implementations; adding one
  speculatively is just indirection. Built it alongside the cache for that reason.
- **ffill is range-dependent**, so the cache stores *raw* closes and lets
  `load_prices` forward-fill once — otherwise a cached sub-range could differ
  from a freshly-fetched one.
- Measured win: a warm cache served a year of 2-ticker data in ~0.002s vs ~1.1s
  cold (~600x), identical values — and zero yfinance calls (the reliability win).

### Problems

- DuckDB's DATE round-trip changes the index datetime *resolution* (us/ns); it's
  irrelevant to numeric work, so the cache-vs-upstream equality test compares
  with `check_index_type=False`.
- Lazy DuckDB connection (open on first use, not in `__init__`) was needed so
  importing the API / constructing the source creates no stray cache file —
  keeps the no-network test suite clean.

### Next

- `PriceSource` for delisted tickers / point-in-time data (kills survivorship bias).
- Parameter-sensitivity sweep + Monte Carlo resampling — see [[ideas.md]].
