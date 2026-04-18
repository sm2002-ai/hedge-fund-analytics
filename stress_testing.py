import numpy as np
import pandas as pd
import yfinance as yf
from config import PORTFOLIO_TICKERS, BENCHMARK


CRISIS_PERIODS = {
    "GFC 2008-09": ("2008-09-01", "2009-03-31"),
    "COVID Crash 2020": ("2020-02-19", "2020-03-23"),
    "Dotcom Bust 2000-02": ("2000-03-10", "2002-10-09"),
    "2022 Rate Shock": ("2022-01-01", "2022-10-13"),
}

SHOCK_SCENARIOS = {
    "Market -5%": -0.05,
    "Market -10%": -0.10,
    "Market -15%": -0.15,
    "Market -20%": -0.20,
}


def run_historical_crisis(portfolio_returns: pd.Series, portfolio_weights: dict) -> dict:
    """Replay historical crisis periods using benchmark drawdown as a proxy."""
    results = {}
    spy = yf.download(BENCHMARK, start="1999-01-01", progress=False, auto_adjust=True)
    spy_ret = spy["Close"].pct_change().dropna()

    for name, (start, end) in CRISIS_PERIODS.items():
        period_spy = spy_ret.loc[start:end]
        if period_spy.empty:
            results[name] = {"spy_return": None, "portfolio_est_return": None}
            continue

        spy_total = float((1 + period_spy).prod() - 1)

        portfolio_period = portfolio_returns.loc[start:end]
        if len(portfolio_period) > 5:
            port_total = float((1 + portfolio_period).prod() - 1)
        else:
            avg_beta = 1.0
            port_total = avg_beta * spy_total

        results[name] = {
            "spy_return": spy_total,
            "portfolio_return": port_total,
            "relative_performance": port_total - spy_total,
            "start": start,
            "end": end,
        }
    return results


def run_single_day_shocks(portfolio_beta: float = 1.0) -> dict:
    """Estimate single-day portfolio loss for market shocks using portfolio beta."""
    results = {}
    for name, shock in SHOCK_SCENARIOS.items():
        port_shock = shock * portfolio_beta
        results[name] = {
            "market_shock": shock,
            "portfolio_impact": port_shock,
            "portfolio_impact_pct": port_shock * 100,
        }
    return results


def run_rate_hike_scenario(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, rate_hike_bps: int = 100
) -> dict:
    """Estimate impact of a rate hike shock using historical rate sensitivity."""
    import yfinance as yf

    tlt = yf.download("TLT", start=portfolio_returns.index[0].strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    tlt_ret = tlt["Close"].pct_change().dropna()

    aligned = pd.concat([portfolio_returns, tlt_ret], axis=1).dropna()
    if len(aligned) < 30:
        rate_sensitivity = -0.5
    else:
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        rate_sensitivity = cov[0, 1] / cov[1, 1]

    tlt_move = -rate_hike_bps / 10000 * 17.0
    portfolio_impact = rate_sensitivity * tlt_move

    return {
        "rate_hike_bps": rate_hike_bps,
        "rate_sensitivity_to_tlt": float(rate_sensitivity),
        "estimated_tlt_move": float(tlt_move),
        "estimated_portfolio_impact": float(portfolio_impact),
        "estimated_portfolio_impact_pct": float(portfolio_impact * 100),
    }


def run_recession_scenario(portfolio_returns: pd.Series, earnings_decline_pct: float = 0.20) -> dict:
    """Estimate portfolio impact under an earnings recession scenario."""
    avg_pe_sensitivity = 0.8
    price_decline = -(earnings_decline_pct * avg_pe_sensitivity)

    historical_vol = portfolio_returns.std() * np.sqrt(252)
    zscore_decline = price_decline / (historical_vol / np.sqrt(252) + 1e-9)

    return {
        "assumed_earnings_decline": earnings_decline_pct,
        "estimated_price_impact": price_decline,
        "estimated_price_impact_pct": price_decline * 100,
        "zscore_equivalent": float(zscore_decline),
        "note": "Based on P/E mean-reversion and earnings pass-through assumption",
    }


def run_all_stress_tests(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                          portfolio_weights: dict, portfolio_beta: float = 1.0) -> dict:
    """Run the full suite of stress tests."""
    return {
        "historical_crises": run_historical_crisis(portfolio_returns, portfolio_weights),
        "single_day_shocks": run_single_day_shocks(portfolio_beta),
        "rate_hike_100bps": run_rate_hike_scenario(portfolio_returns, benchmark_returns, 100),
        "rate_hike_200bps": run_rate_hike_scenario(portfolio_returns, benchmark_returns, 200),
        "recession_mild": run_recession_scenario(portfolio_returns, 0.10),
        "recession_severe": run_recession_scenario(portfolio_returns, 0.25),
    }
