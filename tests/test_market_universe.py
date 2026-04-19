"""Tests for market_universe."""

from __future__ import annotations

import pandas as pd

from market_universe import (
    MACRO_TICKERS,
    SECTOR_ETFS,
    _FALLBACK,
    fetch_sp500_universe,
    filter_universe,
)


def test_sector_etfs_cover_all_gics():
    # 11 GICS sectors.
    assert len(SECTOR_ETFS) == 11


def test_macro_tickers_non_empty():
    assert "SPY" in MACRO_TICKERS
    assert "^VIX" in MACRO_TICKERS


def test_fallback_has_all_sectors():
    sectors = {s for _, s in _FALLBACK}
    assert len(sectors) == 11


def test_fetch_sp500_fallback(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("market_universe.requests.get", boom)
    df = fetch_sp500_universe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ticker", "sector"]
    assert len(df) > 0


def test_filter_universe_by_sector():
    universe = pd.DataFrame(_FALLBACK, columns=["ticker", "sector"])
    filtered = filter_universe(universe, sectors=["Financials"])
    assert (filtered["sector"] == "Financials").all()


def test_filter_universe_none_returns_all():
    universe = pd.DataFrame(_FALLBACK, columns=["ticker", "sector"])
    assert filter_universe(universe, None).equals(universe)
