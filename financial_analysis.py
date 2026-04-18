#!/usr/bin/env python3
"""
Financial analysis: fetch fundamentals via yfinance, score and rank each ticker,
compute optimal portfolio weights, and rewrite portfolio.csv.

Scoring dimensions (each 0-100, then averaged):
  - Profitability: gross/operating/net margin, ROE, ROA
  - Value:         P/E, forward P/E, P/B, EV/EBITDA (lower = better)
  - Growth:        revenue YoY growth
  - Risk:          beta (lower = better), debt/equity (lower = better)
"""
import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

TICKERS = ["TENB", "VRNS", "LOAR", "NOW", "PANW", "CPNG", "WVE"]

# Original shares and cost basis — preserved from the real portfolio
ORIGINAL_DATA = {
    "TENB": {"shares": 30,  "cost_basis": 16.9833, "sector": "Cybersecurity"},
    "VRNS": {"shares": 16,  "cost_basis": 22.2169, "sector": "Cybersecurity"},
    "LOAR": {"shares": 5,   "cost_basis": 56.9000, "sector": "Aerospace & Defense"},
    "NOW":  {"shares": 2,   "cost_basis": 83.8200, "sector": "Software & Cloud"},
    "PANW": {"shares": 1,   "cost_basis": 175.1400,"sector": "Cybersecurity"},
    "CPNG": {"shares": 14,  "cost_basis": 20.6579, "sector": "E-Commerce"},
    "WVE":  {"shares": 19,  "cost_basis": 7.5176,  "sector": "Healthcare & Biotech"},
}

SCORE_WEIGHTS = {
    "profitability": 0.35,
    "value":         0.25,
    "growth":        0.25,
    "risk":          0.15,
}


def _safe(val, default=np.nan):
    """Return val if it is a finite number, else default."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def fetch_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}

    # --- price / valuation ---
    price           = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap      = _safe(info.get("marketCap"))
    pe              = _safe(info.get("trailingPE"))
    fwd_pe          = _safe(info.get("forwardPE"))
    pb              = _safe(info.get("priceToBook"))
    ev_ebitda       = _safe(info.get("enterpriseToEbitda"))

    # --- profitability ---
    gross_margin    = _safe(info.get("grossMargins"))
    op_margin       = _safe(info.get("operatingMargins"))
    net_margin      = _safe(info.get("profitMargins"))
    roe             = _safe(info.get("returnOnEquity"))
    roa             = _safe(info.get("returnOnAssets"))

    # --- growth ---
    rev_growth      = _safe(info.get("revenueGrowth"))

    # --- risk ---
    beta            = _safe(info.get("beta"))
    de_ratio        = _safe(info.get("debtToEquity"))

    # --- 52-week range ---
    high_52w        = _safe(info.get("fiftyTwoWeekHigh"))
    low_52w         = _safe(info.get("fiftyTwoWeekLow"))

    # --- free cash flow ---
    fcf             = _safe(info.get("freeCashflow"))

    sector          = info.get("sector", ORIGINAL_DATA[ticker]["sector"])
    industry        = info.get("industry", "")

    return {
        "ticker": ticker,
        "price": price,
        "market_cap": market_cap,
        "pe": pe,
        "fwd_pe": fwd_pe,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roa": roa,
        "rev_growth": rev_growth,
        "beta": beta,
        "de_ratio": de_ratio,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "fcf": fcf,
        "sector": sector,
        "industry": industry,
    }


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank values 0-100; NaN stays NaN."""
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return (ranked * 100).clip(0, 100)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add score columns and composite score to df."""
    s = df.copy()

    # --- Profitability (higher = better) ---
    prof_components = []
    for col in ["gross_margin", "op_margin", "net_margin", "roe", "roa"]:
        r = percentile_rank(s[col], higher_is_better=True)
        prof_components.append(r)
    s["score_profitability"] = pd.concat(prof_components, axis=1).mean(axis=1)

    # --- Value (lower ratio = better, so invert) ---
    # Replace negative valuation ratios with NaN — negative P/E or EV/EBITDA
    # means the company has no earnings/EBITDA and is NOT "cheap".
    val_components = []
    for col in ["pe", "fwd_pe", "pb", "ev_ebitda"]:
        cleaned = s[col].where(s[col] > 0)   # NaN out non-positive values
        r = percentile_rank(cleaned, higher_is_better=False)
        val_components.append(r)
    s["score_value"] = pd.concat(val_components, axis=1).mean(axis=1)

    # --- Growth (higher = better) ---
    s["score_growth"] = percentile_rank(s["rev_growth"], higher_is_better=True)

    # --- Risk (lower beta and D/E = better) ---
    risk_components = []
    for col in ["beta", "de_ratio"]:
        r = percentile_rank(s[col], higher_is_better=False)
        risk_components.append(r)
    s["score_risk"] = pd.concat(risk_components, axis=1).mean(axis=1)

    # --- Composite score ---
    s["composite_score"] = (
        SCORE_WEIGHTS["profitability"] * s["score_profitability"].fillna(50) +
        SCORE_WEIGHTS["value"]         * s["score_value"].fillna(50) +
        SCORE_WEIGHTS["growth"]        * s["score_growth"].fillna(50) +
        SCORE_WEIGHTS["risk"]          * s["score_risk"].fillna(50)
    )

    return s


def classify_rating(row, mean_score: float, std_score: float) -> str:
    if row["composite_score"] >= mean_score + 0.5 * std_score:
        return "Overweight"
    elif row["composite_score"] <= mean_score - 0.5 * std_score:
        return "Underweight"
    else:
        return "Equal Weight"


def compute_weights(scores: pd.Series, min_weight: float = 0.02) -> pd.Series:
    """Softmax-style weight allocation with a floor so no stock gets zero weight."""
    shifted = scores - scores.min() + 1.0   # +1 ensures minimum stock gets > 0
    weights = shifted / shifted.sum()
    # Apply minimum weight floor and re-normalize
    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()
    return weights.round(4)


def main():
    print("=" * 60)
    print("  Financial Analysis — Scoring & Weight Optimization")
    print("=" * 60)

    records = []
    for ticker in TICKERS:
        print(f"  Fetching {ticker}...", end="", flush=True)
        try:
            rec = fetch_fundamentals(ticker)
            records.append(rec)
            print(f" price=${rec['price']:.2f}  rev_growth={rec['rev_growth']}")
        except Exception as e:
            print(f" ERROR: {e}")
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")

    print("\n[Scoring]")
    df = score_dataframe(df)

    mean_s = df["composite_score"].mean()
    std_s  = df["composite_score"].std()
    df["rating"] = df.apply(lambda r: classify_rating(r, mean_s, std_s), axis=1)

    weights = compute_weights(df["composite_score"])
    df["weight"] = weights

    # Print ranking table
    ranking = df[["price","composite_score","score_profitability","score_value",
                  "score_growth","score_risk","rating","weight"]].sort_values(
                      "composite_score", ascending=False)
    print("\n  Score Summary:")
    print(f"  {'Ticker':<6} {'Score':>6} {'Prof':>6} {'Value':>6} {'Growth':>6} {'Risk':>6}  {'Rating':<13} {'Weight':>7}")
    print("  " + "-" * 70)
    for ticker, row in ranking.iterrows():
        print(f"  {ticker:<6} {row['composite_score']:>6.1f} "
              f"{row['score_profitability']:>6.1f} {row['score_value']:>6.1f} "
              f"{row['score_growth']:>6.1f} {row['score_risk']:>6.1f}  "
              f"{row['rating']:<13} {row['weight']:>7.4f}")

    # Rebuild portfolio.csv
    rows = []
    for ticker in TICKERS:
        orig = ORIGINAL_DATA[ticker]
        # Always use our own sector names (yfinance uses generic ones like "Technology")
        sector = orig["sector"]
        weight = float(df.loc[ticker, "weight"]) if ticker in df.index else 1 / len(TICKERS)
        rows.append({
            "ticker":     ticker,
            "shares":     orig["shares"],
            "cost_basis": orig["cost_basis"],
            "sector":     sector,
            "weight":     round(weight, 4),
        })

    portfolio_df = pd.DataFrame(rows)
    # Normalize to ensure weights sum to 1
    portfolio_df["weight"] = (portfolio_df["weight"] / portfolio_df["weight"].sum()).round(4)
    # Adjust rounding residual on the highest-scored stock
    diff = 1.0 - portfolio_df["weight"].sum()
    top_idx = df["composite_score"].idxmax()
    portfolio_df.loc[portfolio_df["ticker"] == top_idx, "weight"] += round(diff, 4)

    portfolio_df.to_csv("portfolio.csv", index=False)
    print(f"\n  portfolio.csv updated. Weights sum = {portfolio_df['weight'].sum():.4f}")

    # Also print the detailed fundamentals
    print("\n  Fundamentals:")
    detail_cols = ["price","pe","fwd_pe","pb","ev_ebitda","gross_margin",
                   "op_margin","net_margin","roe","roa","rev_growth","beta","de_ratio"]
    for ticker in TICKERS:
        if ticker not in df.index:
            continue
        r = df.loc[ticker]
        print(f"\n  {ticker}:")
        for c in detail_cols:
            val = r.get(c, np.nan)
            if pd.isna(val):
                print(f"    {c}: N/A")
            elif c in ["gross_margin","op_margin","net_margin","roe","roa","rev_growth"]:
                print(f"    {c}: {val*100:.1f}%")
            else:
                print(f"    {c}: {val:.4g}")

    print("\nDone — portfolio.csv is ready.\n")


if __name__ == "__main__":
    main()
