# Decisions

Technical choices and the reasoning behind them. See [[devlog.md]] for when
each decision was made and [[concepts.md]] for background on the ideas.

## uv for Python + dependency management

Initially used pip + venv because `uv` wasn't installed and Python 3.11 wasn't
either (only 3.9.16 and 3.12.12 were available via pyenv), so the repo was
pinned to 3.12.12 as a "3.11+"-compatible fallback. Once `uv` was installed,
switched over: `uv python install 3.11` gave an actual 3.11.15 without
touching pyenv, `uv python pin 3.11` set `.python-version`, and
`uv sync --extra dev` replaced the manual `venv` + `pip install -e ".[dev]"`
flow. Two concrete wins over pip + venv: (1) `uv sync` resolves and writes
`uv.lock`, so the dev environment is reproducible across machines instead of
floating on whatever `pip install` happens to resolve that day; (2)
`uv python install` sidesteps pyenv entirely — no shim/PATH ordering issues,
and project-local Python versions don't require a global pyenv version
switch. `uv.lock` is committed (matches the project's "reproducibility" goal
from the top-level constraints); `.venv/` stays gitignored as before. Day to
day: `uv run pytest`, `uv run ruff check .` instead of activating the venv
manually.

## src-layout package (`src/backtester/...`)

Keeps the installable package separate from the repo root (tests, docs,
config), so `import backtester` only ever resolves to the installed package,
never to a same-named file accidentally sitting at the repo root picked up
via the current working directory.

## Signals own their shift, not the caller

`momentum_signal` calls `.shift(1)` internally rather than returning a raw
score and trusting the backtest engine (or the user) to shift it before use.
The whole point of the no-lookahead rule is that it's easy to forget — making
the *signal function itself* lookahead-safe means every caller gets the
guarantee for free, and it's the thing covered directly by
`tests/test_momentum.py`. `momentum_score` (the unshifted version) is still
exported for cases that genuinely need the as-of-today value (e.g. computing
the score that *will become* the next period's signal), but it's named
distinctly so it can't be mistaken for the tradable signal.

## Cost model as a frozen dataclass with a single `trade_costs` method

`CostModel(fixed_fee=..., proportional_bps=..., ...)` groups the four cost
parameters into one typed, immutable object instead of four loose function
arguments threaded through the engine. `frozen=True` because a cost
assumption shouldn't mutate mid-backtest. All four cost components are
computed vectorized (no Python loops) over the position-changes series so
this scales the same way the rest of the engine does.

## Market impact: quadratic-in-size, not a calibrated model

`impact = market_impact_coef * traded_size**2 * price`. Real market-impact
models (e.g. Almgren-Chriss) are calibrated against order-book depth and
participation rate, which is out of scope for an educational backtester. The
quadratic form is kept because it captures the right *qualitative* behavior
(impact cost grows faster than the trade size itself) without pretending to
be more accurate than it is — the docstring on `CostModel` says so
explicitly.

## Loader fetches per-ticker rather than batching `yf.download(group_by="ticker")`

`yf.Ticker(ticker).history(...)` is called once per ticker instead of one
batched `yf.download` call. This trades a bit of speed for two things that
matter more here: (1) one bad/delisted ticker just gets skipped with a
warning instead of corrupting or failing the whole batch, and (2) it's
trivially mockable in tests (`tests/test_loader.py` swaps in a `FakeTicker`)
without needing to fake yfinance's batched MultiIndex column layout, which
shifts shape depending on how many tickers are requested.

## US equities by default, ASX behind a flag

The demo and API now default to US tickers because they are the path of least
resistance for an educational project: no exchange suffix (yfinance treats US as
the default), deeper and cleaner history, far more liquid names (so the loader's
gap-fill rarely fires), and almost every quant tutorial/paper uses US equities
so the concepts line up. The capability was already there — `load_prices` had an
`asx` flag — so this was a defaults change plus `--asx` on `run_demo.py`, not a
loader rewrite. ASX stays a first-class option, just not the default.

## Engine works in weights, not share counts

`run_backtest` interprets positions as portfolio *weights* (fraction of capital
per asset). This keeps the whole simulation unit-consistent: portfolio return is
a weighted average of asset returns, costs are charged on `|Δweight|`, and the
equity curve is dimensionless growth scaled by starting capital. Share-count
accounting would force the engine to carry a capital base and per-share lot
sizing — more realism than an educational vectorized engine needs, and harder to
reason about. The trade-off (documented on `run_backtest`) is that summing raw
±1 signals across N assets implies N× leverage, so callers equal-weight
(`positions / n_assets`) when they want ~100% gross exposure.

## Two cost methods: `trade_costs` (currency) and `trade_cost_fraction` (weights)

Rather than rewrite the existing, tested `CostModel`, the weight-based engine got
a second method. `trade_cost_fraction` charges only the linear bps terms
(proportional + slippage) on `|Δweight|`, because those have a natural
fraction-of-notional meaning. `fixed_fee` (a flat currency amount) and
`market_impact` (currency, price-dependent) have no clean weight-space meaning
without an assumed capital base and per-share price, so they stay in
`trade_costs` for share/notional work. Keeping both, instead of forcing one
model to do both jobs, avoids smuggling hidden assumptions into the equity curve.

## Engine does not shift positions; signals already did

The no-lookahead shift lives in the signal layer (`momentum_signal`,
`pairs_positions` both `.shift(1)`), so `run_backtest` treats
`positions.loc[t]` as already tradable and combines it with the return realized
over period `t`. If the engine also shifted, positions would be double-lagged
and results silently wrong. This mirrors the existing "signals own their shift"
decision and is asserted directly in `tests/test_engine.py`.

## Pairs hedge ratio: in-sample by default, walk-forward for honesty

`engle_granger` fit on the whole sample is in-sample — the hedge ratio has seen
the entire price path, so a backtest using it is optimistic. `pairs_positions`
accepts an explicit `hedge_ratio=` precisely so it can be fit on a trailing
window and driven out-of-sample by `walk_forward`. The in-sample default is kept
for convenience and analysis, with the caveat stated loudly in the docstring,
rather than pretending the full-sample fit is tradable.

## Walk-forward is signal-agnostic (takes a `strategy` callable)

`walk_forward` doesn't import any signal; it takes a
`(train_prices, test_prices) -> positions` callable. This lets the same harness
validate momentum (warm-up only) and pairs (re-fit the hedge ratio on train)
without the validation layer depending on the signal layer. The harness owns the
window split; the strategy owns being lookahead-safe inside the test window.

## FastAPI entrypoint at `backtester.api.app:app`; Docker on the uv image

The API is a thin layer over the existing functions (`load_prices` ->
`momentum_positions` -> `run_backtest` -> `summarize`), so it adds no new core
logic and no new dependencies (fastapi/uvicorn/pydantic were already declared,
so `uv.lock` is untouched and CI's `--locked` check stays green). The Dockerfile
uses the official `ghcr.io/astral-sh/uv` image and `uv sync --locked --no-dev`
so the container builds the exact environment CI tests against. The API test
monkeypatches `load_prices` to stay no-network like the rest of the suite.

## Full CI/CD: type-check in CI, publish a smoke-tested image in CD

CI gained a `mypy` step, plus a `src/backtester/py.typed` marker — without the
marker mypy skips the package entirely ("cannot be type checked due to missing
py.typed marker"), so the step would be a no-op. CD is a separate workflow
(`.github/workflows/cd.yml`) that builds the Docker image, **boots it and checks
`/health` before publishing** — a broken Dockerfile or start command fails the
pipeline rather than reaching users — then pushes to GHCR. GHCR was chosen over
Docker Hub because the built-in `GITHUB_TOKEN` can push to it with
`packages: write`, so there are no external secrets to manage for an educational
repo. CD triggers on `main`, on `v*` tags (semver-tagged images), and
`workflow_dispatch` so it can be exercised from a feature branch before merging.
Deploying the published image to a live host is left as a documented manual step
— no hosting provider or credentials are assumed in the repo.

## ruff + pytest, configured in `pyproject.toml`

Both configured under `[tool.pytest.ini_options]` and `[tool.ruff]` rather
than separate config files, to keep all project config in one place. Ruff
line-length is 100 (not the default 88) — long type-hinted pandas function
signatures (`pd.DataFrame | pd.Series`) hit 88 quickly without adding any
real complexity.
