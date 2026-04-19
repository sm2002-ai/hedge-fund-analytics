"""Stock screens for market-wide opportunity ranking.

All screens take a wide price DataFrame (index=dates, columns=tickers) and
return a tidy DataFrame with one row per ticker. Scores are z-scored so they
can be combined into a composite without any single factor dominating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def momentum_score(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Classic 12M-1M momentum: total return from t-lookback to t-skip.

    Skipping the most recent month is standard in momentum research to avoid
    short-term mean reversion.
    """
    if len(prices) < lookback + 1:
        lookback = max(len(prices) - skip - 1, 1)
    end = prices.iloc[-skip - 1] if skip > 0 and len(prices) > skip else prices.iloc[-1]
    start = prices.iloc[-lookback - 1] if len(prices) > lookback else prices.iloc[0]
    return (end / start - 1).rename("momentum")


def volatility_score(prices: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """Annualized realized volatility over the lookback window (lower = better)."""
    rets = _daily_returns(prices).tail(lookback)
    return (rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)).rename("volatility")


def sharpe_score(prices: pd.DataFrame, lookback: int = 252, rf_daily: float = 0.0) -> pd.Series:
    """Annualized Sharpe over the lookback window."""
    rets = _daily_returns(prices).tail(lookback)
    excess = rets - rf_daily
    mean = excess.mean()
    std = excess.std().replace(0, np.nan)
    sharpe = (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe.fillna(0.0).rename("sharpe")


def max_drawdown_score(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Max drawdown over the lookback window (negative number; closer to 0 = better)."""
    window = prices.tail(lookback)
    cum = window / window.iloc[0]
    drawdown = (cum / cum.cummax() - 1).min()
    return drawdown.rename("max_drawdown")


def trend_score(prices: pd.DataFrame, short: int = 50, long: int = 200) -> pd.Series:
    """Simple trend gauge: 50D SMA / 200D SMA - 1. >0 means uptrend."""
    short_sma = prices.tail(short).mean()
    long_sma = prices.tail(long).mean() if len(prices) >= long else prices.mean()
    trend = (short_sma / long_sma - 1).fillna(0.0)
    return trend.rename("trend")


def build_screener_frame(prices: pd.DataFrame, sectors: pd.Series | None = None) -> pd.DataFrame:
    """Compute all signals and a composite rank for each ticker.

    The composite is a simple equal-weighted z-score blend:
      +momentum +sharpe +trend  -volatility  -|max_drawdown|

    Signs are chosen so that larger composite = more attractive long.
    """
    prices = prices.dropna(how="all", axis=1)
    prices = prices.ffill(limit=5).dropna(how="any", axis=1)

    mom = momentum_score(prices)
    vol = volatility_score(prices)
    shr = sharpe_score(prices)
    dd = max_drawdown_score(prices)
    tr = trend_score(prices)

    df = pd.concat([mom, vol, shr, dd, tr], axis=1)
    df["momentum_z"] = _zscore(df["momentum"])
    df["sharpe_z"] = _zscore(df["sharpe"])
    df["trend_z"] = _zscore(df["trend"])
    df["volatility_z"] = _zscore(df["volatility"])
    df["drawdown_z"] = _zscore(df["max_drawdown"].abs())

    df["composite"] = (
        df["momentum_z"]
        + df["sharpe_z"]
        + df["trend_z"]
        - df["volatility_z"]
        - df["drawdown_z"]
    )

    if sectors is not None:
        df = df.join(sectors.rename("sector"), how="left")
    df = df.sort_values("composite", ascending=False)
    df.index.name = "ticker"
    return df


def top_n(
    screener: pd.DataFrame, n: int = 10, min_price: float = 5.0,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top N tickers by composite score, filtered for liquidity/price sanity."""
    out = screener.copy()
    if prices is not None and min_price > 0:
        last = prices.iloc[-1]
        keep = last[last >= min_price].index
        out = out.loc[out.index.intersection(keep)]
    return out.head(n)


def momentum_leaders(screener: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return screener.sort_values("momentum", ascending=False).head(n)


def low_vol_high_sharpe(screener: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Names with below-median vol AND above-median Sharpe."""
    med_vol = screener["volatility"].median()
    med_shr = screener["sharpe"].median()
    filtered = screener[
        (screener["volatility"] < med_vol) & (screener["sharpe"] > med_shr)
    ]
    return filtered.sort_values("sharpe", ascending=False).head(n)


def value_bounce(screener: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Beaten-down names (bottom quartile of trailing momentum) that are
    turning up (positive 50/200 trend). Often captures early mean-reversion."""
    mom_cutoff = screener["momentum"].quantile(0.25)
    filtered = screener[(screener["momentum"] <= mom_cutoff) & (screener["trend"] > 0)]
    return filtered.sort_values("trend", ascending=False).head(n)
