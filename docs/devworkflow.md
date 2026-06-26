# Dev workflow

How work actually moves through this repo. See [[decisions.md]] for *why* the
tooling is set up this way and [[devlog.md]] for the running history.

## Environment

`uv` owns Python and dependencies (see [[decisions.md]]):

```bash
uv sync --extra dev      # Python 3.11 (from .python-version) + deps, against uv.lock
```

Day to day, prefix commands with `uv run` instead of activating the venv:

```bash
uv run pytest -q                         # tests (no network: yfinance is mocked)
uv run ruff check .                      # lint
uv run mypy                              # type-check (needs src/backtester/py.typed)
uv run python scripts/run_demo.py        # end-to-end demo (hits live yfinance)
uv run uvicorn backtester.api.app:app --reload   # the HTTP API + /docs
```

## Branch naming

Single long-lived branch is `main` (tracks `origin`). All other work happens on
short-lived branches named `<type>/<short-kebab-description>`:

| Prefix      | Use for                                             |
|-------------|-----------------------------------------------------|
| `feat/`     | a new capability (e.g. `feat/backtest-engine`)      |
| `fix/`      | a bug fix (e.g. `fix/api-date-validation`)          |
| `docs/`     | docs-only changes (e.g. `docs/concepts-metrics`)    |
| `refactor/` | restructuring with no behavior change               |
| `chore/`    | tooling, deps, CI, config                           |

Keep the description a few kebab-case words; one logical change per branch.

## Commit messages

Short imperative subject (optionally `<type>: ` prefixed, matching the branch),
then a body explaining *what changed and why* when it isn't obvious. Group
related changes into one commit rather than one commit per file.

## Pull requests + CI/CD

```bash
git checkout -b feat/<name>
# ...work, committing as you go...
git push -u origin feat/<name>
gh pr create        # or open the PR in the GitHub UI
```

**CI** (`.github/workflows/ci.yml`) runs on every PR (and push to `main`): it
installs the locked deps with `uv sync --extra dev --locked` (fails if `uv.lock`
is stale), then `ruff check .`, then `mypy`, then `pytest -q`. Get CI green
before merging. Run those three locally first so the PR is green on the first try.

**CD** (`.github/workflows/cd.yml`) runs on push to `main`, on `v*` tags, or via
manual dispatch: it builds the Docker image, smoke-tests that the container
answers `/health`, and publishes it to GHCR
(`ghcr.io/<owner>/quant-backtesting-engine`). Cut a release by tagging:

```bash
git tag v0.1.0 && git push origin v0.1.0     # -> semver-tagged image on GHCR
```

Deploying that image to a live host (Fly.io, Render, etc.) is a manual step;
the pipeline's job is to always have a known-good, runnable image published.

## Keep the docs current

This project treats `docs/` as a first-class artifact. When a change introduces
a new idea or a non-obvious choice, update in the same branch:

- [[devlog.md]] — what was built / learned / hit problems on, dated.
- [[concepts.md]] — plain-language explanation of any new concept.
- [[decisions.md]] — the reasoning behind a technical choice.
