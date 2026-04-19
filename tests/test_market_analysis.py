"""Tests for market_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_analysis import (
    factor_performance,
    macro_snapshot,
    market_regime,
    market_risk_snapshot,
    sector_leaderboard,
)


@pytest.fixture
def idx():
    return pd.bdate_range("2023-01-02", periods=300)


@pytest.fixture
def spy_uptrend(idx):
    # Deterministic monotonic uptrend so SMA ordering is guaranteed.
    return pd.Series(np.linspace(100, 200, len(idx)), index=idx)


@pytest.fixture
def spy_downtrend(idx):
    return pd.Series(np.linspace(200, 100, len(idx)), index=idx)


def test_regime_uptrend(spy_uptrend):
    vix = pd.Series(np.full(300, 14.0), index=spy_uptrend.index)
    regime = market_regime(spy_uptrend, vix)
    assert regime["trend"] == "uptrend"
    assert regime["vol_regime"] == "low"
    assert regime["bias"] == "risk-on"


def test_regime_downtrend(spy_downtrend):
    vix = pd.Series(np.full(300, 40.0), index=spy_downtrend.index)
    regime = market_regime(spy_downtrend, vix)
    assert regime["trend"] == "downtrend"
    assert regime["vol_regime"] == "stress"
    assert regime["bias"] == "risk-off"


def test_regime_no_vix(spy_uptrend):
    regime = market_regime(spy_uptrend, None)
    assert regime["vol_regime"] == "unknown"
    assert regime["vix"] is None


def test_regime_has_return_windows(spy_uptrend):
    regime = market_regime(spy_uptrend)
    assert "spy_1m" in regime
    assert "spy_3m" in regime
    assert "spy_ytd" in regime


def test_macro_snapshot(idx):
    rng = np.random.default_rng(2)
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))),
        "GLD": 180 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(idx))),
    }, index=idx)
    labels = {"SPY": "S&P 500", "GLD": "Gold", "MISSING": "Not in data"}
    snap = macro_snapshot(prices, labels)
    tickers = [r["ticker"] for r in snap]
    assert "SPY" in tickers
    assert "GLD" in tickers
    assert "MISSING" not in tickers
    for row in snap:
        assert set(row.keys()) == {"ticker", "label", "last", "r_1m", "r_3m", "r_ytd"}


def test_sector_leaderboard(idx):
    rng = np.random.default_rng(3)
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))),
        "XLK": 150 * np.cumprod(1 + rng.normal(0.0008, 0.012, len(idx))),
        "XLF": 40 * np.cumprod(1 + rng.normal(0.0003, 0.011, len(idx))),
    }, index=idx)
    sector_map = {"XLK": "Technology", "XLF": "Financials"}
    board = sector_leaderboard(prices, sector_map)
    assert len(board) == 2
    for row in board:
        assert "r_1m" in row and "sharpe" in row and "beta" in row
        assert np.isfinite(row["beta"])


def test_sector_leaderboard_requires_spy(idx):
    prices = pd.DataFrame({"XLK": np.arange(1, len(idx) + 1)}, index=idx)
    with pytest.raises(ValueError):
        sector_leaderboard(prices, {"XLK": "Tech"})


def test_market_risk_snapshot(idx):
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
    snap = market_risk_snapshot(rets)
    assert snap["var_95"] > 0
    assert snap["cvar_95"] >= snap["var_95"]
    assert snap["max_drawdown"] <= 0
    assert snap["annualized_vol"] > 0


def test_market_risk_snapshot_empty():
    assert market_risk_snapshot(pd.Series(dtype=float)) == {}


def test_factor_performance():
    idx = pd.bdate_range("2024-01-02", periods=120)
    rng = np.random.default_rng(5)
    ff = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0005, 0.01, len(idx)),
        "SMB": rng.normal(0.0, 0.006, len(idx)),
        "HML": rng.normal(0.0, 0.006, len(idx)),
        "RMW": rng.normal(0.0, 0.005, len(idx)),
        "CMA": rng.normal(0.0, 0.005, len(idx)),
    }, index=idx)
    perf = factor_performance(ff)
    assert len(perf) == 5
    for row in perf:
        assert row["factor"] in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
        assert np.isfinite(row["r_1m"])
        assert np.isfinite(row["r_3m"])
        assert np.isfinite(row["r_ytd"])
