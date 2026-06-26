from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data.sources import CachedPriceSource, PriceSource, YFinanceSource


class RecordingSource:
    """A fake PriceSource that serves slices of a fixed frame and counts calls."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        *,
        price_field: str = "Close",
    ) -> pd.DataFrame:
        self.calls += 1
        cols = [t for t in tickers if t in self.frame.columns]
        if not cols:
            raise ValueError(f"No price data returned for any of tickers={tickers!r}")
        mask = self.frame.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.frame.index <= pd.Timestamp(end)
        return self.frame.loc[mask, cols].copy()


@pytest.fixture
def frame() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=260)  # ~ one year of trading days
    rng = np.random.default_rng(0)
    data = {
        t: 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=len(idx))))
        for t in ("AAPL", "MSFT")
    }
    return pd.DataFrame(data, index=idx)


def test_recording_source_satisfies_protocol(frame: pd.DataFrame) -> None:
    assert isinstance(RecordingSource(frame), PriceSource)
    assert isinstance(YFinanceSource(), PriceSource)


def test_second_identical_request_is_served_from_cache(frame: pd.DataFrame, tmp_path) -> None:
    upstream = RecordingSource(frame)
    cache = CachedPriceSource(upstream, tmp_path / "prices.duckdb")

    first = cache.get_prices(["AAPL", "MSFT"], "2022-02-01", "2022-06-30")
    calls_after_first = upstream.calls
    second = cache.get_prices(["AAPL", "MSFT"], "2022-02-01", "2022-06-30")

    assert upstream.calls == calls_after_first  # no new upstream fetch
    pd.testing.assert_frame_equal(first, second)


def test_cached_values_match_upstream(frame: pd.DataFrame, tmp_path) -> None:
    upstream = RecordingSource(frame)
    cache = CachedPriceSource(upstream, tmp_path / "prices.duckdb")

    cached = cache.get_prices(["AAPL"], "2022-02-01", "2022-03-31")
    direct = upstream.get_prices(["AAPL"], "2022-02-01", "2022-03-31")
    # Same dates and values; the DuckDB round-trip can change the datetime *resolution*
    # of the index (e.g. us vs ns), which is irrelevant to downstream numeric work.
    pd.testing.assert_series_equal(
        cached["AAPL"], direct["AAPL"], check_freq=False, check_index_type=False
    )


def test_cache_persists_across_instances(frame: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "prices.duckdb"
    warm = CachedPriceSource(RecordingSource(frame), path)
    warm.get_prices(["AAPL"], "2022-02-01", "2022-06-30")

    # A brand-new instance + a source that would error if called: must serve from disk.
    cold = CachedPriceSource(RecordingSource(pd.DataFrame()), path)
    result = cold.get_prices(["AAPL"], "2022-02-01", "2022-06-30")
    assert not result.empty


def test_subrange_of_cached_span_needs_no_fetch(frame: pd.DataFrame, tmp_path) -> None:
    upstream = RecordingSource(frame)
    cache = CachedPriceSource(upstream, tmp_path / "prices.duckdb")

    cache.get_prices(["AAPL"], "2022-02-01", "2022-08-31")
    calls = upstream.calls
    cache.get_prices(["AAPL"], "2022-03-01", "2022-06-30")  # strictly inside cached span
    assert upstream.calls == calls


def test_extending_range_triggers_one_more_fetch(frame: pd.DataFrame, tmp_path) -> None:
    upstream = RecordingSource(frame)
    cache = CachedPriceSource(upstream, tmp_path / "prices.duckdb")

    cache.get_prices(["AAPL"], "2022-02-01", "2022-04-30")
    calls = upstream.calls
    extended = cache.get_prices(["AAPL"], "2022-02-01", "2022-09-30")  # past cached end
    assert upstream.calls == calls + 1
    assert extended.index.max() >= pd.Timestamp("2022-09-01")


def test_all_unknown_tickers_raise(frame: pd.DataFrame, tmp_path) -> None:
    cache = CachedPriceSource(RecordingSource(frame), tmp_path / "prices.duckdb")
    with pytest.raises(ValueError, match="No price data"):
        cache.get_prices(["NOPE"], "2022-02-01", "2022-03-31")
