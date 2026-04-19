"""Market-regime, macro and sector-leaderboard analytics.

All functions are pure (take price data, return dicts/DataFrames). yfinance
is called only by the orchestrator in market_report.py so these functions
stay testable without network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _window_returns(prices: pd.Series, days: int) -> float:
    if len(prices) < 2:
        return 0.0
    window = prices.tail(min(days + 1, len(prices)))
    return float(window.iloc[-1] / window.iloc[0] - 1)


def market_regime(spy_prices: pd.Series, vix_prices: pd.Series | None = None) -> dict:
    """Classify current market regime from SPY trend + VIX level."""
    if len(spy_prices) < 200:
        short_sma = spy_prices.mean()
        long_sma = short_sma
    else:
        short_sma = spy_prices.tail(50).mean()
        long_sma = spy_prices.tail(200).mean()

    spot = float(spy_prices.iloc[-1])
    trend = "uptrend" if spot > short_sma > long_sma else (
        "downtrend" if spot < short_sma < long_sma else "sideways"
    )

    vix_level = float(vix_prices.iloc[-1]) if vix_prices is not None and len(vix_prices) else None
    if vix_level is None:
        vol_regime = "unknown"
    elif vix_level < 15:
        vol_regime = "low"
    elif vix_level < 25:
        vol_regime = "normal"
    elif vix_level < 35:
        vol_regime = "elevated"
    else:
        vol_regime = "stress"

    # Risk-on if uptrend + non-stress vol, risk-off if downtrend or stress vol.
    if trend == "uptrend" and vol_regime in ("low", "normal"):
        bias = "risk-on"
    elif trend == "downtrend" or vol_regime == "stress":
        bias = "risk-off"
    else:
        bias = "neutral"

    return {
        "spot": spot,
        "sma_50": float(short_sma),
        "sma_200": float(long_sma),
        "trend": trend,
        "vix": vix_level,
        "vol_regime": vol_regime,
        "bias": bias,
        "spy_1m": _window_returns(spy_prices, 21),
        "spy_3m": _window_returns(spy_prices, 63),
        "spy_ytd": _ytd_return(spy_prices),
    }


def _ytd_return(prices: pd.Series) -> float:
    if prices.empty:
        return 0.0
    last_date = prices.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        return _window_returns(prices, 252)
    year_start = pd.Timestamp(last_date.year, 1, 1)
    sliced = prices[prices.index >= year_start]
    if len(sliced) < 2:
        return 0.0
    return float(sliced.iloc[-1] / sliced.iloc[0] - 1)


def macro_snapshot(prices: pd.DataFrame, labels: dict[str, str]) -> list[dict]:
    """Build a macro dashboard: name, last, 1M, 3M, YTD return per ticker."""
    out = []
    for ticker, label in labels.items():
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue
        out.append({
            "ticker": ticker,
            "label": label,
            "last": float(series.iloc[-1]),
            "r_1m": _window_returns(series, 21),
            "r_3m": _window_returns(series, 63),
            "r_ytd": _ytd_return(series),
        })
    return out


def sector_leaderboard(prices: pd.DataFrame, sector_map: dict[str, str]) -> list[dict]:
    """Per-sector ETF: return, vol, Sharpe, 1M momentum, beta to SPY."""
    if "SPY" not in prices.columns:
        raise ValueError("sector_leaderboard requires SPY in prices")

    spy = prices["SPY"].pct_change().dropna()
    rows = []
    for ticker, name in sector_map.items():
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < 30:
            continue
        rets = s.pct_change().dropna()
        aligned = pd.concat([rets, spy], axis=1, join="inner").dropna()
        aligned.columns = ["sec", "spy"]
        cov = np.cov(aligned["sec"], aligned["spy"])
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else float("nan")

        vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if rets.std() else 0.0

        rows.append({
            "ticker": ticker,
            "sector": name,
            "r_1m": _window_returns(s, 21),
            "r_3m": _window_returns(s, 63),
            "r_ytd": _ytd_return(s),
            "vol": vol,
            "sharpe": sharpe,
            "beta": beta,
        })
    rows.sort(key=lambda r: r["r_ytd"], reverse=True)
    return rows


def factor_performance(ff5: pd.DataFrame) -> list[dict]:
    """Trailing factor returns: 1M, 3M, YTD for each FF5 factor."""
    out = []
    for factor in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        if factor not in ff5.columns:
            continue
        series = ff5[factor].dropna()
        if series.empty:
            continue
        cum = (1 + series).cumprod()
        out.append({
            "factor": factor,
            "r_1m": float(cum.iloc[-1] / cum.iloc[max(-22, -len(cum))] - 1) if len(cum) >= 22 else 0.0,
            "r_3m": float(cum.iloc[-1] / cum.iloc[max(-64, -len(cum))] - 1) if len(cum) >= 64 else 0.0,
            "r_ytd": _ytd_cumret(series),
        })
    return out


def _ytd_cumret(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    last_date = returns.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        return float((1 + returns).prod() - 1)
    year_start = pd.Timestamp(last_date.year, 1, 1)
    sliced = returns[returns.index >= year_start]
    return float((1 + sliced).prod() - 1) if not sliced.empty else 0.0


def market_risk_snapshot(spy_returns: pd.Series, confidence: float = 0.95) -> dict:
    """VaR/CVaR/drawdown on the broad market index."""
    if spy_returns.empty:
        return {}
    var = float(-np.percentile(spy_returns.dropna(), (1 - confidence) * 100))
    tail = spy_returns[spy_returns <= -var]
    cvar = float(-tail.mean()) if not tail.empty else var
    cum = (1 + spy_returns).cumprod()
    drawdown = float((cum / cum.cummax() - 1).min())
    return {
        "var_95": var,
        "cvar_95": cvar,
        "max_drawdown": drawdown,
        "annualized_vol": float(spy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
    }
