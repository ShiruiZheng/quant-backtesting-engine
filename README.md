# quant-backtesting-engine

Vectorized Python backtesting engine for ASX equities with transaction costs, walk-forward validation, risk metrics, FastAPI, Docker, and CI.

Educational/research project — not for live trading.

## Project layout

```
src/backtester/
  data/        # loader.py: fetches prices from yfinance (US by default; .AX for ASX, gap-fill)
  core/        # returns.py: simple/log/cumulative returns
               # costs.py:   CostModel (fixed fee, bps, slippage, market impact; + weight-space costs)
               # engine.py:  run_backtest -> positions x returns - costs -> equity curve (BacktestResult)
               # metrics.py: Sharpe, Sortino, Calmar, max drawdown, CAGR, VaR, win/hit rate, vol
  signals/     # momentum.py: momentum_score -> momentum_signal (shifted) -> momentum_positions
               # pairs.py:    Engle-Granger cointegration + z-score pairs/stat-arb signal
  validation/  # walkforward.py: rolling train/test out-of-sample harness
  api/         # app.py: FastAPI service (GET /health, POST /backtest)
tests/         # mirrors src/, one test file per module, plus no-lookahead checks
scripts/       # run_demo.py: manual end-to-end run against live data (see below)
docs/          # devlog.md, concepts.md, decisions.md
Dockerfile, docker-compose.yml   # containerized API
```

The pipeline is now end-to-end: fetch prices -> build a lagged signal
(momentum or pairs) -> simulate it into an **equity curve** with transaction
costs -> score it with **risk/performance metrics** -> stress it with
**walk-forward** out-of-sample validation -> optionally serve it over **HTTP**.
The honest test is walk-forward, not the in-sample run: on the demo data the
in-sample momentum Sharpe (~1.3) collapses out-of-sample (~0.06), which is the
whole point of having the harness.

## Quickstart

```bash
uv sync --extra dev      # installs Python 3.11 (pinned in .python-version) + deps, writes/reads uv.lock
uv run pytest -q         # run the test suite (no network: yfinance is mocked in tests)
uv run ruff check .      # lint
uv run python scripts/run_demo.py                                  # US default: AAPL MSFT GOOGL AMZN NVDA
uv run python scripts/run_demo.py --tickers AAPL MSFT NVDA --start 2021-01-01 --lookback 40
uv run python scripts/run_demo.py --asx --tickers BHP CBA CSL     # ASX instead of US
```

`run_demo.py` is the guided end-to-end tour. It hits yfinance for real US
prices (use `--asx` for Australian listings), then runs, in order:
the momentum **next-day target positions**, a full **backtest** (equity curve +
metrics), **walk-forward** out-of-sample metrics, and an **Engle-Granger pairs**
fit + backtest on the first two tickers. There's no caching/storage layer yet —
every run re-fetches from yfinance.

## Run the API

```bash
uv run uvicorn backtester.api.app:app --reload     # then open http://127.0.0.1:8000/docs
curl -s -X POST http://127.0.0.1:8000/backtest \
  -H 'content-type: application/json' \
  -d '{"tickers":["AAPL","MSFT","GOOGL","AMZN","NVDA"],"start":"2022-01-01","lookback":60}'
```

`GET /health` is a liveness check; `POST /backtest` fetches prices, runs an
equal-weight momentum backtest, and returns metrics, the next-period target
positions, and the equity curve as JSON. The interactive Swagger UI at `/docs`
is the easiest way to try it.

## Run with Docker

```bash
docker compose up --build      # serves the API on http://127.0.0.1:8000
```

The image installs the exact locked dependencies (`uv sync --locked --no-dev`),
so it builds the same environment CI tests against.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every PR: installs
the pinned Python + locked deps via `uv sync --extra dev --locked` (fails if
`uv.lock` is out of date), then `ruff check .` and `pytest -q`.

## Branching

Single `main` branch today, tracking `origin` on GitHub. For new work, branch
off main (`git checkout -b feat/<name>`), push, and open a PR — CI will run
lint + tests on the PR before it's safe to merge.
