"""Tests for market_screener."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_screener import (
    build_screener_frame,
    low_vol_high_sharpe,
    max_drawdown_score,
    momentum_leaders,
    momentum_score,
    sharpe_score,
    top_n,
    trend_score,
    value_bounce,
    volatility_score,
)


@pytest.fixture
def price_panel():
    idx = pd.bdate_range("2022-01-03", periods=300)
    rng = np.random.default_rng(42)
    tickers = ["UP", "DOWN", "FLAT", "VOL", "CALM"]
    out = {}
    for t in tickers:
        if t == "UP":
            rets = rng.normal(0.001, 0.01, len(idx))
        elif t == "DOWN":
            rets = rng.normal(-0.0008, 0.012, len(idx))
        elif t == "FLAT":
            rets = rng.normal(0.0, 0.005, len(idx))
        elif t == "VOL":
            rets = rng.normal(0.0002, 0.03, len(idx))
        else:
            rets = rng.normal(0.0005, 0.004, len(idx))
        out[t] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(out, index=idx)


def test_momentum_score_shape(price_panel):
    mom = momentum_score(price_panel)
    assert set(mom.index) == set(price_panel.columns)
    assert mom.name == "momentum"


def test_momentum_score_sign(price_panel):
    mom = momentum_score(price_panel)
    # UP should beat DOWN.
    assert mom["UP"] > mom["DOWN"]


def test_volatility_score_positive(price_panel):
    vol = volatility_score(price_panel)
    assert (vol > 0).all()
    # VOL series should be more volatile than CALM.
    assert vol["VOL"] > vol["CALM"]


def test_sharpe_score_finite(price_panel):
    shr = sharpe_score(price_panel)
    assert np.isfinite(shr).all()


def test_max_drawdown_non_positive(price_panel):
    dd = max_drawdown_score(price_panel)
    assert (dd <= 0).all()


def test_trend_score(price_panel):
    tr = trend_score(price_panel)
    assert tr.name == "trend"
    # Upward trend should register positive.
    assert tr["UP"] > tr["DOWN"]


def test_build_screener_frame_columns(price_panel):
    df = build_screener_frame(price_panel)
    for col in ["momentum", "volatility", "sharpe", "max_drawdown", "trend", "composite"]:
        assert col in df.columns
    assert df.index.name == "ticker"


def test_build_screener_frame_sorted(price_panel):
    df = build_screener_frame(price_panel)
    composites = df["composite"].tolist()
    assert composites == sorted(composites, reverse=True)


def test_build_screener_frame_with_sectors(price_panel):
    sectors = pd.Series({t: "Tech" for t in price_panel.columns})
    df = build_screener_frame(price_panel, sectors=sectors)
    assert "sector" in df.columns
    assert (df["sector"] == "Tech").all()


def test_top_n_respects_min_price(price_panel):
    df = build_screener_frame(price_panel)
    ideas = top_n(df, n=3, min_price=0.01, prices=price_panel)
    assert len(ideas) <= 3


def test_top_n_filters_penny_stocks(price_panel):
    df = build_screener_frame(price_panel)
    # Set last price of UP below min_price to ensure filtering works.
    mutated = price_panel.copy()
    mutated.iloc[-1, mutated.columns.get_loc("UP")] = 1.0
    ideas = top_n(df, n=5, min_price=5.0, prices=mutated)
    assert "UP" not in ideas.index


def test_momentum_leaders_sorted(price_panel):
    df = build_screener_frame(price_panel)
    leaders = momentum_leaders(df, n=3)
    assert len(leaders) == 3
    assert leaders["momentum"].tolist() == sorted(leaders["momentum"].tolist(), reverse=True)


def test_low_vol_high_sharpe(price_panel):
    df = build_screener_frame(price_panel)
    picks = low_vol_high_sharpe(df, n=5)
    med_vol = df["volatility"].median()
    med_shr = df["sharpe"].median()
    assert (picks["volatility"] < med_vol).all()
    assert (picks["sharpe"] > med_shr).all()


def test_value_bounce(price_panel):
    df = build_screener_frame(price_panel)
    picks = value_bounce(df, n=5)
    mom_cutoff = df["momentum"].quantile(0.25)
    assert (picks["momentum"] <= mom_cutoff).all()
    assert (picks["trend"] > 0).all()


def test_build_screener_handles_constant_columns():
    idx = pd.bdate_range("2022-01-03", periods=300)
    prices = pd.DataFrame({
        "CONST": np.full(len(idx), 100.0),
        "WALK": 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, len(idx))),
    }, index=idx)
    df = build_screener_frame(prices)
    assert np.isfinite(df["composite"]).all()
