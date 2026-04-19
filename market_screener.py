#!/usr/bin/env python3
"""
Market Screener — Multibagger Edition
Screens for 10x stock candidates using "The Alchemy of Multibagger Stocks" framework.
"""
import os
import sys
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

os.makedirs("output", exist_ok=True)

# ── Universe of candidates ──────────────────────────────────────────────────
# Small/mid-cap growth + value candidates across sectors
SCREEN_UNIVERSE = [
    # Tech / Software
    "CRDO", "ASAN", "DDOG", "GTLB", "BILL", "ZI", "DOCS", "SEMR",
    "ALKT", "ESMT", "WEAV", "MGNI", "PAYO", "SMMT", "SQSP", "RAMP",
    # Cybersecurity
    "TENB", "VRNS", "QLYS", "RDWR", "IRTC", "EVTC", "STNE",
    # Industrials / Defense
    "LOAR", "KTOS", "RKLB", "SPIR", "ACHR", "JOBY",
    # Healthcare / Biotech
    "RXRX", "IONS", "EXAS", "PAC", "KNSL", "KYMR",
    # Financials / Fintech
    "SOFI", "AFRM", "UPST", "LPRO", "CURO", "NRDS", "FLYW",
    # Consumer / E-commerce
    "CPNG", "CELH", "VITL", "ELF", "SKIN", "BYON", "WEBR",
    # Energy / Commodities
    "CDEV", "MTDR", "BATL", "DINO", "CLNE",
    # Real assets / REITs
    "STWD", "BXMT", "ACRE", "BRSP",
    # Large cap for alpha comparison
    "NOW", "PANW", "CRWD", "SNOW", "PLTR",
]

# Industry median P/E reference table (approximations)
INDUSTRY_PE = {
    "Technology":          28.0,
    "Software—Application": 35.0,
    "Cybersecurity":       32.0,
    "Semiconductors":      25.0,
    "Healthcare":          22.0,
    "Biotechnology":       35.0,
    "Industrials":         20.0,
    "Financial Services":  15.0,
    "Consumer Cyclical":   18.0,
    "Consumer Defensive":  22.0,
    "Energy":              12.0,
    "Real Estate":         18.0,
    "Communication Services": 22.0,
    "Default":             22.0,
}


def _safe(val, default=None):
    """Return val if it's a real number, else default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch all fundamentals needed for multibagger screening via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # ── Price & Market Cap ────────────────────────────────────────────
        price = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
        market_cap = _safe(info.get("marketCap"))

        # ── Valuation ────────────────────────────────────────────────────
        trailing_pe   = _safe(info.get("trailingPE"))
        forward_pe    = _safe(info.get("forwardPE"))
        ps_ratio      = _safe(info.get("priceToSalesTrailing12Months"))
        pb_ratio      = _safe(info.get("priceToBook"))

        # ── Profitability ─────────────────────────────────────────────────
        roe           = _safe(info.get("returnOnEquity"))
        roa           = _safe(info.get("returnOnAssets"))
        gross_margin  = _safe(info.get("grossMargins"))
        op_margin     = _safe(info.get("operatingMargins"))
        profit_margin = _safe(info.get("profitMargins"))
        ebitda        = _safe(info.get("ebitda"))

        # ── Growth ───────────────────────────────────────────────────────
        rev_growth    = _safe(info.get("revenueGrowth"))
        earn_growth   = _safe(info.get("earningsGrowth"))

        # ── FCF ──────────────────────────────────────────────────────────
        fcf           = _safe(info.get("freeCashflow"))
        op_cashflow   = _safe(info.get("operatingCashflow"))

        # Derive FCF yield = FCF / market_cap
        fcf_yield = None
        if fcf is not None and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap

        # ── Balance Sheet ─────────────────────────────────────────────────
        total_debt    = _safe(info.get("totalDebt"))
        total_equity  = _safe(info.get("bookValue"))
        shares        = _safe(info.get("sharesOutstanding"))
        debt_equity   = None
        if total_debt is not None and total_equity is not None and shares and total_equity * shares > 0:
            debt_equity = total_debt / (total_equity * shares)
        elif total_debt is not None and market_cap and market_cap > 0:
            debt_equity = total_debt / market_cap  # proxy

        # ── ROCE / ROIC proxy ─────────────────────────────────────────────
        # ROCE ≈ EBIT / (Total Assets - Current Liabilities)
        # Use returnOnAssets as a proxy when full balance sheet not available
        roce = _safe(info.get("returnOnAssets"))  # proxy; real ROCE needs full BS

        # ── Technical / Moving averages ───────────────────────────────────
        price_50d  = _safe(info.get("fiftyDayAverage"))
        price_200d = _safe(info.get("twoHundredDayAverage"))
        avg_vol_10d = _safe(info.get("averageVolume10days"))
        avg_vol_3m  = _safe(info.get("averageVolume"))

        # Volume trend: ratio of 10d avg to 3m avg
        vol_trend = None
        if avg_vol_10d and avg_vol_3m and avg_vol_3m > 0:
            vol_trend = avg_vol_10d / avg_vol_3m

        # ── Sector / Industry ─────────────────────────────────────────────
        sector   = info.get("sector", "Default")
        industry = info.get("industry", "")
        short_name = info.get("shortName", ticker)

        # ── Total Assets / Revenue for EBITDA-to-asset comparison ─────────
        total_assets  = _safe(info.get("totalAssets"))
        revenue       = _safe(info.get("totalRevenue"))

        return {
            "ticker":       ticker,
            "name":         short_name,
            "sector":       sector,
            "industry":     industry,
            "price":        price,
            "market_cap":   market_cap,
            "trailing_pe":  trailing_pe,
            "forward_pe":   forward_pe,
            "ps_ratio":     ps_ratio,
            "pb_ratio":     pb_ratio,
            "roe":          roe,
            "roa":          roa,
            "roce":         roce,
            "gross_margin": gross_margin,
            "op_margin":    op_margin,
            "profit_margin":profit_margin,
            "ebitda":       ebitda,
            "total_assets": total_assets,
            "revenue":      revenue,
            "rev_growth":   rev_growth,
            "earn_growth":  earn_growth,
            "fcf":          fcf,
            "fcf_yield":    fcf_yield,
            "op_cashflow":  op_cashflow,
            "total_debt":   total_debt,
            "debt_equity":  debt_equity,
            "price_50d":    price_50d,
            "price_200d":   price_200d,
            "vol_trend":    vol_trend,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def score_multibagger(f: dict) -> tuple[float, dict, list]:
    """
    Score a stock on the Multibagger framework.

    Returns:
        (multibagger_score 0-100, tier_breakdown dict, fail_reasons list)

    Tier 1 (must-have filters — hard disqualify if all 5 fail badly):
        FCF Yield > 5%
        Forward P/E < 12x OR P/E < 60% of industry avg
        ROCE/ROIC > 15%
        Revenue Growth > 15% YoY
        Debt/Equity < 0.5

    Tier 2 (scoring, max 28 pts):
        ROE > 15%               +5
        Gross Margin > 40%      +5
        Op Margin > 15%         +5
        EBITDA > Asset Growth   +5  (proxy: EBITDA / Total Assets > 8%)
        Small-cap < $5B         +3
        P/S < 1.0               +3
        Small vs TAM            +2

    Tier 3 (technical, max 15 pts):
        50d MA trending up      +5
        Price > 200d MA         +5
        Volume increasing       +5
    """
    tier1 = {}
    tier2 = {}
    tier3 = {}
    fail_reasons = []

    # ── Tier 1 ────────────────────────────────────────────────────────────
    # FCF Yield
    fcf_yield = f.get("fcf_yield")
    if fcf_yield is not None:
        tier1["fcf_yield_pass"] = fcf_yield > 0.05
        if fcf_yield <= 0.05:
            fail_reasons.append(f"FCF yield {fcf_yield*100:.1f}% < 5%")
    else:
        tier1["fcf_yield_pass"] = False
        fail_reasons.append("FCF yield unavailable")

    # Valuation vs industry
    industry_pe = INDUSTRY_PE.get(f.get("sector", "Default"), INDUSTRY_PE["Default"])
    pe_threshold = min(12.0, industry_pe * 0.6)
    fwd_pe = f.get("forward_pe")
    trl_pe = f.get("trailing_pe")
    pe_check = False
    if fwd_pe is not None and fwd_pe > 0:
        pe_check = fwd_pe < pe_threshold
    elif trl_pe is not None and trl_pe > 0:
        pe_check = trl_pe < (industry_pe * 0.6)
    tier1["valuation_pass"] = pe_check
    if not pe_check:
        fpe_str = f"{fwd_pe:.1f}" if fwd_pe else "N/A"
        fail_reasons.append(f"Valuation not cheap (fwd PE={fpe_str}, threshold={pe_threshold:.1f})")

    # ROCE > 15%
    roce = f.get("roce")
    if roce is not None:
        tier1["roce_pass"] = roce > 0.15
        if roce <= 0.15:
            fail_reasons.append(f"ROCE/ROA {roce*100:.1f}% < 15%")
    else:
        tier1["roce_pass"] = False
        fail_reasons.append("ROCE unavailable")

    # Revenue growth > 15%
    rev_growth = f.get("rev_growth")
    if rev_growth is not None:
        tier1["rev_growth_pass"] = rev_growth > 0.15
        if rev_growth <= 0.15:
            fail_reasons.append(f"Rev growth {rev_growth*100:.1f}% < 15%")
    else:
        tier1["rev_growth_pass"] = False
        fail_reasons.append("Revenue growth unavailable")

    # Debt/Equity < 0.5
    de = f.get("debt_equity")
    if de is not None:
        tier1["debt_equity_pass"] = de < 0.5
        if de >= 0.5:
            fail_reasons.append(f"D/E {de:.2f} >= 0.5")
    else:
        tier1["debt_equity_pass"] = True  # benefit of doubt if unknown

    tier1_passes = sum(1 for v in tier1.values() if v)
    tier1_score = (tier1_passes / 5) * 57  # 57 pts max for tier 1

    # ── Tier 2 ────────────────────────────────────────────────────────────
    t2 = 0.0

    roe = f.get("roe")
    if roe is not None and roe > 0.15:
        t2 += 5
        tier2["roe"] = True
    else:
        tier2["roe"] = False

    gross_margin = f.get("gross_margin")
    if gross_margin is not None and gross_margin > 0.40:
        t2 += 5
        tier2["gross_margin"] = True
    else:
        tier2["gross_margin"] = False

    op_margin = f.get("op_margin")
    if op_margin is not None and op_margin > 0.15:
        t2 += 5
        tier2["op_margin"] = True
    else:
        tier2["op_margin"] = False

    # EBITDA > Asset Growth proxy: EBITDA/Total Assets > 8%
    ebitda = f.get("ebitda")
    total_assets = f.get("total_assets")
    if ebitda and total_assets and total_assets > 0 and ebitda / total_assets > 0.08:
        t2 += 5
        tier2["ebitda_asset_ratio"] = True
    else:
        tier2["ebitda_asset_ratio"] = False

    mc = f.get("market_cap")
    if mc and mc < 5e9:
        t2 += 3
        tier2["small_cap"] = True
    else:
        tier2["small_cap"] = False

    ps = f.get("ps_ratio")
    if ps is not None and 0 < ps < 1.0:
        t2 += 3
        tier2["ps_ratio"] = True
    else:
        tier2["ps_ratio"] = False

    # TAM proxy: small market cap relative to revenue implies large addressable market
    rev = f.get("revenue")
    if mc and rev and rev > 0 and mc / rev < 3.0:
        t2 += 2
        tier2["small_vs_tam"] = True
    else:
        tier2["small_vs_tam"] = False

    tier2_score = t2  # max 28 pts

    # ── Tier 3 ────────────────────────────────────────────────────────────
    t3 = 0.0
    price = f.get("price")
    price_50d = f.get("price_50d")
    price_200d = f.get("price_200d")
    vol_trend = f.get("vol_trend")

    # 50d MA trending up (price > 50d MA)
    if price and price_50d and price > price_50d:
        t3 += 5
        tier3["above_50d"] = True
    else:
        tier3["above_50d"] = False

    # Price > 200d MA
    if price and price_200d and price > price_200d:
        t3 += 5
        tier3["above_200d"] = True
    else:
        tier3["above_200d"] = False

    # Volume increasing
    if vol_trend and vol_trend > 1.1:
        t3 += 5
        tier3["vol_trend"] = True
    else:
        tier3["vol_trend"] = False

    tier3_score = (t3 / 15) * 15

    raw_score = tier1_score + tier2_score + tier3_score  # max ~100
    multibagger_score = min(100.0, raw_score)

    breakdown = {
        "tier1": tier1, "tier1_score": round(tier1_score, 1),
        "tier2": tier2, "tier2_score": round(tier2_score, 1),
        "tier3": tier3, "tier3_score": round(tier3_score, 1),
        "tier1_passes": tier1_passes,
    }
    return round(multibagger_score, 1), breakdown, fail_reasons


_SPY_RETURNS_CACHE: dict = {}


def _get_spy_returns(lookback: int = 252) -> pd.Series:
    """Fetch SPY returns once and cache for the process lifetime."""
    if lookback not in _SPY_RETURNS_CACHE:
        start = (datetime.date.today() - datetime.timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
        raw = yf.download("SPY", start=start, progress=False, auto_adjust=True)["Close"]
        raw = raw.dropna().tail(lookback + 1)
        _SPY_RETURNS_CACHE[lookback] = raw.pct_change().dropna()
    return _SPY_RETURNS_CACHE[lookback]


def compute_alpha_score(ticker: str, lookback: int = 252) -> float:
    """Compute a simple alpha score: annualized excess return vs SPY (0-100 scale)."""
    try:
        spy_ret = _get_spy_returns(lookback)
        start = (datetime.date.today() - datetime.timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"]
        if raw.empty:
            return 50.0
        raw = raw.dropna().tail(lookback + 1)
        stock_ret = raw.pct_change().dropna()
        # Align on common dates
        aligned = pd.concat([stock_ret, spy_ret], axis=1, join="inner")
        if aligned.empty or aligned.shape[0] < 20:
            return 50.0
        aligned.columns = ["stock", "spy"]
        stock_ann = float((1 + aligned["stock"].mean()) ** 252 - 1)
        spy_ann   = float((1 + aligned["spy"].mean()) ** 252 - 1)
        alpha = stock_ann - spy_ann
        return float(np.clip(50 + alpha * 100, 0, 100))
    except Exception:
        return 50.0


def compute_original_composite(f: dict) -> float:
    """
    Original composite score (momentum + quality hybrid, 0-100).
    Simple blend of growth, profitability, and valuation signals.
    """
    score = 50.0

    # Revenue growth bonus
    rg = f.get("rev_growth")
    if rg is not None:
        score += min(20, rg * 100)

    # Profitability
    om = f.get("op_margin")
    if om is not None:
        score += min(10, om * 50)

    # Valuation (reward cheapness)
    pe = f.get("forward_pe") or f.get("trailing_pe")
    if pe and 0 < pe < 30:
        score += 10
    elif pe and pe < 50:
        score += 5

    # FCF quality
    fy = f.get("fcf_yield")
    if fy and fy > 0:
        score += min(10, fy * 100)

    return float(np.clip(score, 0, 100))


def screen_multibaggers(universe: list = None, verbose: bool = True) -> pd.DataFrame:
    """
    Full multibagger screening pipeline.

    Returns DataFrame ranked by composite score:
        40% Multibagger Score + 30% Alpha Score + 30% Original Composite
    """
    if universe is None:
        universe = SCREEN_UNIVERSE

    if verbose:
        print(f"\n[Screener] Scanning {len(universe)} tickers for multibagger candidates...")

    results = []
    for i, ticker in enumerate(universe):
        if verbose and (i % 10 == 0):
            print(f"  [{i+1}/{len(universe)}] Processing {ticker}...")
        f = fetch_fundamentals(ticker)
        if "error" in f:
            continue

        mb_score, breakdown, fail_reasons = score_multibagger(f)
        alpha_score = compute_alpha_score(ticker)
        orig_score  = compute_original_composite(f)

        composite = (0.40 * mb_score) + (0.30 * alpha_score) + (0.30 * orig_score)

        results.append({
            "ticker":           ticker,
            "name":             f.get("name", ticker),
            "sector":           f.get("sector", ""),
            "industry":         f.get("industry", ""),
            "price":            f.get("price"),
            "market_cap_bn":    (f.get("market_cap") or 0) / 1e9,
            "fcf_yield_pct":    (f.get("fcf_yield") or 0) * 100,
            "forward_pe":       f.get("forward_pe"),
            "trailing_pe":      f.get("trailing_pe"),
            "ps_ratio":         f.get("ps_ratio"),
            "roe_pct":          (f.get("roe") or 0) * 100,
            "roce_pct":         (f.get("roce") or 0) * 100,
            "gross_margin_pct": (f.get("gross_margin") or 0) * 100,
            "op_margin_pct":    (f.get("op_margin") or 0) * 100,
            "rev_growth_pct":   (f.get("rev_growth") or 0) * 100,
            "debt_equity":      f.get("debt_equity"),
            "above_200d":       f.get("price_200d") and f.get("price") and f["price"] > f["price_200d"],
            "tier1_passes":     breakdown["tier1_passes"],
            "multibagger_score":mb_score,
            "alpha_score":      round(alpha_score, 1),
            "orig_score":       round(orig_score, 1),
            "composite_score":  round(composite, 1),
            "_fundamentals":    f,
            "_breakdown":       breakdown,
            "_fail_reasons":    fail_reasons,
        })

        time.sleep(0.15)  # rate-limit yfinance calls

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def build_investment_thesis(row: pd.Series) -> str:
    """Build a per-stock investment thesis using the Multibagger framework."""
    f   = row["_fundamentals"]
    bd  = row["_breakdown"]
    t   = row["ticker"]
    nm  = row["name"]

    mc_str = f"${row['market_cap_bn']:.1f}B" if row['market_cap_bn'] > 0 else "N/A"

    # FCF yield narrative
    fy = row["fcf_yield_pct"]
    if fy > 5:
        fcf_txt = (f"FCF yield of **{fy:.1f}%** is above the 5% threshold — this is the "
                   f"single strongest multibagger predictor per the research. The company generates "
                   f"real cash that can be reinvested for compounding growth.")
    elif fy > 2:
        fcf_txt = (f"FCF yield of **{fy:.1f}%** is developing but not yet at the 5% threshold. "
                   f"Watch for improvement as operating leverage kicks in.")
    else:
        fcf_txt = (f"FCF yield of **{fy:.1f}%** is below target — company may be in reinvestment "
                   f"phase. Verify that cash burn is funding durable competitive advantages.")

    # Valuation narrative
    fpe = f.get("forward_pe")
    tpe = f.get("trailing_pe")
    ind_pe = INDUSTRY_PE.get(f.get("sector", "Default"), INDUSTRY_PE["Default"])
    if fpe and 0 < fpe < ind_pe * 0.6:
        val_txt = (f"Trading at **{fpe:.1f}x** forward earnings vs. industry median of ~{ind_pe:.0f}x — "
                   f"a **{(1-fpe/ind_pe)*100:.0f}% discount**. Classic deep value with growth characteristics.")
    elif fpe and 0 < fpe < ind_pe:
        val_txt = (f"Forward P/E of **{fpe:.1f}x** is below the industry median ({ind_pe:.0f}x), "
                   f"suggesting modest undervaluation relative to peers.")
    elif fpe and fpe > 0:
        val_txt = (f"Forward P/E of **{fpe:.1f}x** is above the industry median ({ind_pe:.0f}x). "
                   f"Premium warranted only if growth trajectory justifies it.")
    elif tpe and tpe > 0:
        val_txt = f"Trailing P/E of **{tpe:.1f}x** (forward P/E unavailable)."
    else:
        val_txt = "P/E data unavailable — may be pre-profit growth stage."

    # Growth narrative
    rg = row["rev_growth_pct"]
    if rg > 30:
        growth_txt = f"**{rg:.1f}%** revenue growth — hypergrowth phase. If sustainable, this alone can drive 10x in 5-7 years."
    elif rg > 15:
        growth_txt = f"**{rg:.1f}%** revenue growth — above the 15% multibagger threshold. Compounding at this rate doubles revenue in ~5 years."
    elif rg > 0:
        growth_txt = f"**{rg:.1f}%** revenue growth — below the 15% threshold. Needs acceleration to qualify as multibagger candidate."
    else:
        growth_txt = f"Revenue growth data unavailable or negative ({rg:.1f}%)."

    # Balance sheet
    de = row.get("debt_equity")
    if de is not None and de < 0.5:
        bs_txt = f"**Clean balance sheet** with D/E of {de:.2f} — well below 0.5 threshold. Fortress-like financial position allows aggressive reinvestment."
    elif de is not None:
        bs_txt = f"D/E of {de:.2f} is above 0.5 threshold — elevated leverage warrants monitoring, especially in rising rate environment."
    else:
        bs_txt = "Debt/Equity data unavailable."

    # Tier 1 summary
    t1_pass = bd.get("tier1_passes", 0)
    tier1_summary = f"Passes **{t1_pass}/5** Tier 1 must-have filters."

    # Scores
    scores_txt = (f"| Metric | Score |\n|--------|-------|\n"
                  f"| Multibagger Score | **{row['multibagger_score']:.0f}/100** |\n"
                  f"| Alpha Score | {row['alpha_score']:.0f}/100 |\n"
                  f"| Original Composite | {row['orig_score']:.0f}/100 |\n"
                  f"| **COMPOSITE** | **{row['composite_score']:.0f}/100** |")

    # TAM / moat note (qualitative)
    sector = f.get("sector", "")
    if "Technology" in sector or "Software" in sector:
        tam_txt = ("Software businesses with recurring revenue models have large, expanding TAMs as digital "
                   "transformation continues. Market cap at this size is small relative to addressable market.")
    elif "Health" in sector or "Bio" in sector:
        tam_txt = ("Healthcare markets are massive and structurally growing. A single successful product "
                   "can generate multi-billion revenue streams, offering enormous upside from current market cap.")
    elif "Defense" in sector or "Aero" in sector:
        tam_txt = ("Defense budgets are rising globally. Early-stage defense tech companies targeting "
                   "next-generation warfare platforms can grow 10x as contracts scale.")
    elif "Financial" in sector:
        tam_txt = ("Fintech is disrupting a multi-trillion dollar incumbents. Niche leaders that capture "
                   "even 1-2% of their addressable market will see significant revenue expansion.")
    else:
        tam_txt = ("The addressable market opportunity, combined with the company's current market cap "
                   f"of {mc_str}, suggests meaningful room for expansion if execution continues.")

    thesis = f"""## {t} — {nm}

**Market Cap:** {mc_str} | **Sector:** {sector} | **Composite Score:** {row['composite_score']:.0f}/100

### Why It's a Multibagger Candidate

{tier1_summary}

#### 1. Free Cash Flow Yield
{fcf_txt}

#### 2. Valuation Discount
{val_txt}

#### 3. Growth Trajectory
{growth_txt}

#### 4. Balance Sheet Strength
{bs_txt}

#### 5. TAM Opportunity
{tam_txt}

### Scorecard
{scores_txt}

---
"""
    return thesis


def save_investment_thesis(df: pd.DataFrame, top_n: int = 15, output_dir: str = "output") -> str:
    """Save investment theses for top N picks to a markdown file."""
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    path = os.path.join(output_dir, f"investment_thesis_{today}.md")

    top = df.head(top_n)

    header = f"""# Investment Thesis Report — Top {top_n} Multibagger Candidates
**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
**Framework:** The Alchemy of Multibagger Stocks

## Executive Summary

This report identifies the top {top_n} stocks ranked by a composite score that blends:
- **40%** Multibagger Score (FCF yield, valuation, ROCE, growth, leverage)
- **30%** Alpha Score (excess return vs SPY)
- **30%** Original Composite Score (momentum & quality blend)

The screening universe of {len(df)} tickers was filtered through three tiers:
- **Tier 1:** Hard fundamental filters (FCF yield, P/E, ROCE, revenue growth, debt)
- **Tier 2:** Quality scoring (ROE, margins, EBITDA efficiency, size, PS ratio)
- **Tier 3:** Technical momentum (moving averages, volume trend)

## Top {top_n} Rankings

| # | Ticker | Name | Composite | MB Score | Alpha | FCF Yield% | Rev Growth% | Tier1 |
|---|--------|------|-----------|----------|-------|------------|-------------|-------|
"""
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        fy  = f"{row['fcf_yield_pct']:.1f}%"
        rg  = f"{row['rev_growth_pct']:.1f}%"
        t1  = f"{row['tier1_passes']}/5"
        header += (f"| {rank} | **{row['ticker']}** | {row['name'][:30]} | "
                   f"{row['composite_score']:.0f} | {row['multibagger_score']:.0f} | "
                   f"{row['alpha_score']:.0f} | {fy} | {rg} | {t1} |\n")

    header += "\n---\n\n# Individual Stock Theses\n\n"

    theses = ""
    for _, row in top.iterrows():
        theses += build_investment_thesis(row)

    footer = f"""
---
*Disclaimer: This report is generated algorithmically for informational purposes only.
It does not constitute investment advice. Past performance and quantitative scores
do not guarantee future returns. Always conduct independent due diligence.*
"""

    content = header + theses + footer
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Screener] Investment thesis saved → {path}")
    return path


def print_screener_results(df: pd.DataFrame, top_n: int = 15):
    """Print top picks table to console."""
    if df.empty:
        print("[Screener] No results.")
        return

    top = df.head(top_n)
    print(f"\n{'='*90}")
    print(f"  TOP {top_n} MULTIBAGGER CANDIDATES — COMPOSITE SCORE RANKING")
    print(f"  (40% Multibagger + 30% Alpha + 30% Original Composite)")
    print(f"{'='*90}")
    header = (f"{'#':<3} {'Ticker':<7} {'Name':<28} {'Comp':>5} {'MB':>5} "
              f"{'Alpha':>6} {'FCF%':>6} {'RevGr%':>7} {'FwdPE':>6} {'T1':>4}")
    print(header)
    print("-" * 90)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        fpe = f"{row['forward_pe']:.1f}" if row['forward_pe'] and row['forward_pe'] > 0 else "N/A"
        fy  = f"{row['fcf_yield_pct']:.1f}"
        rg  = f"{row['rev_growth_pct']:.1f}"
        t1  = f"{row['tier1_passes']}/5"
        nm  = (row['name'] or "")[:27]
        print(f"{rank:<3} {row['ticker']:<7} {nm:<28} {row['composite_score']:>5.1f} "
              f"{row['multibagger_score']:>5.1f} {row['alpha_score']:>6.1f} "
              f"{fy:>6} {rg:>7} {fpe:>6} {t1:>4}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Market Screener — Multibagger Edition")
    print("  Framework: The Alchemy of Multibagger Stocks")
    print("=" * 60)

    df = screen_multibaggers(verbose=True)

    if df.empty:
        print("[ERROR] No results returned from screener.")
        sys.exit(1)

    print_screener_results(df, top_n=15)

    thesis_path = save_investment_thesis(df, top_n=15)

    # Save full results CSV
    today = datetime.date.today().strftime("%Y-%m-%d")
    csv_path = os.path.join("output", f"screener_results_{today}.csv")
    cols_to_save = [c for c in df.columns if not c.startswith("_")]
    df[cols_to_save].to_csv(csv_path, index=True)
    print(f"[Screener] Full results saved → {csv_path}")

    print("\nDone. Top picks ready for portfolio consideration.")
