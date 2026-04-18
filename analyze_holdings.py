#!/usr/bin/env python3
"""
Fetch fundamentals for each portfolio holding, score them, and update
portfolio.csv with optimal weights based on the scores.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf

os.environ.setdefault("MPLBACKEND", "Agg")

PORTFOLIO_CSV = "portfolio.csv"

# ── scoring weights ────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "profitability": 0.30,
    "value":         0.20,
    "growth":        0.25,
    "risk":          0.25,
}


def fetch_fundamentals(ticker: str) -> dict:
    """Pull key fundamentals from yfinance."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")          # decimal, e.g. 0.23
    profit_margin = info.get("profitMargins") # decimal
    op_margin = info.get("operatingMargins")  # decimal
    rev_growth = info.get("revenueGrowth")    # YoY decimal
    eps_growth = info.get("earningsGrowth")   # YoY decimal
    beta = info.get("beta")
    debt_equity = info.get("debtToEquity")    # as reported (often ×100 in yfinance)

    return {
        "pe":           pe,
        "pb":           pb,
        "roe":          roe,
        "profit_margin": profit_margin,
        "op_margin":    op_margin,
        "rev_growth":   rev_growth,
        "eps_growth":   eps_growth,
        "beta":         beta,
        "debt_equity":  debt_equity,
    }


def score_stock(fundamentals: dict) -> dict:
    """
    Produce sub-scores in [0,1] for each dimension, then a weighted total.
    Missing data falls back to a neutral 0.5.
    """
    def safe(val, default=0.5):
        return default if (val is None or not np.isfinite(val)) else float(val)

    # Profitability (higher is better)
    roe_score  = min(max(safe(fundamentals["roe"])  / 0.30, 0), 1)  # 30% ROE → 1.0
    pm_score   = min(max(safe(fundamentals["profit_margin"]) / 0.25, 0), 1)  # 25% margin → 1.0
    om_score   = min(max(safe(fundamentals["op_margin"]) / 0.30, 0), 1)
    profitability = (roe_score + pm_score + om_score) / 3

    # Value (lower P/E and P/B is better; negative means no earnings/negative equity → 0)
    pe_raw = safe(fundamentals["pe"], default=40)
    pe_score = 0.0 if pe_raw <= 0 else 1 - min(max((pe_raw - 10) / 60, 0), 1)

    pb_raw = safe(fundamentals["pb"], default=5)
    pb_score = 0.0 if pb_raw <= 0 else 1 - min(max((pb_raw - 1) / 15, 0), 1)

    value = (pe_score + pb_score) / 2

    # Growth (higher is better)
    rg_score  = min(max(safe(fundamentals["rev_growth"]) / 0.30 + 0.5, 0), 1)  # 0%→0.5, 30%→1.0
    eg_score  = min(max(safe(fundamentals["eps_growth"]) / 0.40 + 0.5, 0), 1)
    growth = (rg_score + eg_score) / 2

    # Risk (lower beta and debt-equity is better)
    beta_raw = safe(fundamentals["beta"], default=1.2)
    beta_score = 1 - min(max((beta_raw - 0.5) / 1.5, 0), 1)  # 0.5→1.0, 2.0→0.0

    de_raw = safe(fundamentals["debt_equity"], default=100)
    # yfinance sometimes gives D/E ×100; normalise
    if de_raw > 20:
        de_raw /= 100
    de_score = 1 - min(max(de_raw / 2.0, 0), 1)  # 0→1.0, 2.0→0.0

    risk = (beta_score + de_score) / 2

    total = (
        SCORE_WEIGHTS["profitability"] * profitability
        + SCORE_WEIGHTS["value"]        * value
        + SCORE_WEIGHTS["growth"]       * growth
        + SCORE_WEIGHTS["risk"]         * risk
    )

    return {
        "profitability": round(profitability, 4),
        "value":         round(value, 4),
        "growth":        round(growth, 4),
        "risk":          round(risk, 4),
        "total_score":   round(total, 4),
    }


def compute_optimal_weights(scores: pd.Series, min_weight: float = 0.05,
                             max_weight: float = 0.40) -> pd.Series:
    """
    Softmax allocation constrained to [min_weight, max_weight] per holding.
    Iteratively clips and re-normalises.
    """
    w = np.exp(scores.values * 5)          # amplify differences
    w = w / w.sum()

    for _ in range(50):
        clipped = np.clip(w, min_weight, max_weight)
        excess = clipped.sum() - 1.0
        if abs(excess) < 1e-9:
            break
        # redistribute excess proportionally among uncapped
        mask = (clipped > min_weight) & (clipped < max_weight)
        if mask.sum() == 0:
            break
        clipped[mask] -= excess * (clipped[mask] / clipped[mask].sum())
        w = clipped

    w = w / w.sum()
    return pd.Series(w, index=scores.index).round(4)


def main():
    df = pd.read_csv(PORTFOLIO_CSV)
    tickers = df["ticker"].tolist()

    print("=" * 60)
    print("  Hedge Fund — Fundamental Analysis & Weight Optimisation")
    print("=" * 60)

    records = []
    for ticker in tickers:
        print(f"\n  Fetching {ticker} …", end=" ", flush=True)
        try:
            fund = fetch_fundamentals(ticker)
            scores = score_stock(fund)
            print(f"score={scores['total_score']:.3f}", end="")
            records.append({"ticker": ticker, **fund, **scores})
        except Exception as exc:
            print(f"ERROR: {exc}")
            records.append({"ticker": ticker, "total_score": 0.5})

    analysis = pd.DataFrame(records).set_index("ticker")

    # Compute optimal weights
    weights = compute_optimal_weights(analysis["total_score"])
    analysis["optimal_weight"] = weights

    # Update portfolio.csv
    weight_map = weights.to_dict()
    df["weight"] = df["ticker"].map(weight_map)
    df.to_csv(PORTFOLIO_CSV, index=False)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  ANALYSIS SUMMARY")
    print("=" * 60)
    display_cols = ["profitability", "value", "growth", "risk", "total_score", "optimal_weight"]
    available = [c for c in display_cols if c in analysis.columns]
    print(analysis[available].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n  KEY FUNDAMENTALS")
    print("=" * 60)
    fund_cols = ["pe", "roe", "profit_margin", "rev_growth", "beta"]
    avail_fund = [c for c in fund_cols if c in analysis.columns]
    fmt = {c: lambda x: f"{x:.2f}" if pd.notna(x) else "N/A" for c in avail_fund}
    print(analysis[avail_fund].to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"))

    print(f"\n  portfolio.csv updated with optimal weights.")
    print("=" * 60)


if __name__ == "__main__":
    main()
