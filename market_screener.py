#!/usr/bin/env python3
"""
Medallion-Style Market Screener
- BETA plays (95% weight): Non-S&P500 stocks — exploit pricing inefficiencies, mean reversion,
  RSI oversold, Bollinger bounces in less-covered small/mid caps
- ALPHA plays (5% weight): Multibagger fundamentals — FCF Yield, P/E, ROCE, Revenue Growth
"""
import os
import sys
import time
import warnings
import re
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PORTFOLIO_SIZE = 1_000_000   # $1M notional
BETA_WEIGHT    = 0.95
ALPHA_WEIGHT   = 0.05
N_BETA_PICKS   = 80
N_ALPHA_PICKS  = 6
CHUNK_SIZE     = 200
MIN_PRICE      = 1.0        # exclude penny stocks
MIN_AVG_VOLUME = 50_000     # minimum 50k avg daily volume

# ----------------------------------------------
# UNIVERSE BUILDING
# ----------------------------------------------

def _wiki_tickers(url, col_hints):
    try:
        # Wikipedia blocks pandas' default UA — fetch with requests first
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text), header=0)
        for t in tables:
            for hint in col_hints:
                col = next((c for c in t.columns if hint.lower() in c.lower()), None)
                if col:
                    raw = t[col].dropna().astype(str)
                    raw = raw.str.replace(r"\.", "-", regex=True).str.strip()
                    raw = raw[raw.str.match(r"^[A-Z]{1,5}(-[A-Z])?$")]
                    return set(raw.tolist())
    except Exception as e:
        print(f"  [warn] {url}: {e}")
    return set()


def get_sp500():
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ["symbol", "ticker"],
    )


def get_nasdaq100():
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        ["ticker", "symbol"],
    )


def get_sp400():
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        ["ticker", "symbol"],
    )


def get_sp600():
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        ["ticker", "symbol"],
    )


def get_russell3000_ishares():
    """Download IWV (iShares Russell 3000 ETF) holdings CSV."""
    urls = [
        "https://www.ishares.com/us/products/239710/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.ishares.com/us/products/239710/",
    }
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                continue
            lines = r.text.splitlines()
            # Find first line that looks like a CSV header
            start = 0
            for i, line in enumerate(lines):
                if re.search(r"ticker|symbol", line, re.I):
                    start = i
                    break
            content = "\n".join(lines[start:])
            df = pd.read_csv(StringIO(content), on_bad_lines="skip")
            col = next((c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()), None)
            if col:
                raw = df[col].dropna().astype(str).str.strip()
                raw = raw[raw.str.match(r"^[A-Z]{1,5}$")]
                tickers = set(raw.tolist())
                if len(tickers) > 500:
                    return tickers
        except Exception as e:
            print(f"  [warn] iShares download: {e}")
    return set()


def get_nasdaq_trader():
    """Fetch all US-listed common stocks from Nasdaq Trader SymDir.
    Covers NASDAQ + NYSE + NYSE American + ARCA + BATS + IEX."""
    sources = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
         "Symbol", True),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
         "ACT Symbol", False),
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    # Security Name classes to reject (warrants, rights, units, preferreds, debt, funds)
    reject_re = re.compile(
        r"\b(warrant|warrants|right|rights|unit|units|preferred|depositary|"
        r"depository|closed end fund|note|notes|bond|bonds|debenture|trust "
        r"preferred|subordinate|convertible)\b",
        re.IGNORECASE,
    )
    tickers = set()
    for url, sym_col, has_fin_status in sources:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                continue
            lines = [ln for ln in r.text.splitlines()
                     if ln and not ln.startswith("File Creation")]
            df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
            if sym_col not in df.columns:
                continue
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]
            if "ETF" in df.columns:
                df = df[df["ETF"] == "N"]
            if has_fin_status and "Financial Status" in df.columns:
                df = df[df["Financial Status"] == "N"]
            if "Security Name" in df.columns:
                df = df[~df["Security Name"].fillna("").str.contains(reject_re)]
            raw = df[sym_col].dropna().astype(str).str.strip()
            raw = raw[raw.str.match(r"^[A-Z]{1,5}$")]
            tickers |= set(raw.tolist())
        except Exception as e:
            print(f"  [warn] Nasdaq Trader {url}: {e}")
    return tickers


def build_universe():
    print("Building ticker universe…")
    sp500     = get_sp500()
    print(f"  S&P 500:           {len(sp500):>4d}")
    nasdaq100 = get_nasdaq100()
    print(f"  NASDAQ 100:        {len(nasdaq100):>4d}")
    sp400     = get_sp400()
    print(f"  S&P MidCap 400:    {len(sp400):>4d}")
    sp600     = get_sp600()
    print(f"  S&P SmallCap 600:  {len(sp600):>4d}")
    russell   = get_russell3000_ishares()
    print(f"  Russell 3000 (IWV):{len(russell):>4d}")
    nasdaq_trader = get_nasdaq_trader()
    print(f"  Nasdaq Trader:     {len(nasdaq_trader):>4d}")

    all_tickers = sp500 | nasdaq100 | sp400 | sp600 | russell | nasdaq_trader
    # Sanitize
    all_tickers = {
        t for t in all_tickers
        if isinstance(t, str) and re.match(r"^[A-Z]{1,5}(-[A-Z])?$", t)
    }
    print(f"  -------------------------")
    print(f"  Total unique:      {len(all_tickers):>4d}")
    indexed = sp500 | nasdaq100 | sp400 | sp600  # ~1,600 tickers — safe scope for fundamentals
    return all_tickers, sp500, indexed


# ----------------------------------------------
# BATCH PRICE DOWNLOAD
# ----------------------------------------------

def _extract_ticker_df(data, ticker):
    """Handle both MultiIndex (multi-ticker) and flat (single-ticker) yf.download output."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker].dropna(how="all")
    else:
        if "Close" in data.columns:
            return data.dropna(how="all")
    return None


def download_price_data(tickers, period="1y"):
    tickers = sorted(tickers)
    results = {}
    n_chunks = (len(tickers) - 1) // CHUNK_SIZE + 1
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        print(f"  Chunk {chunk_num:>2d}/{n_chunks}  ({len(chunk)} tickers)…", end=" ", flush=True)
        try:
            raw = yf.download(
                chunk,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            ok = 0
            for t in chunk:
                df = _extract_ticker_df(raw, t)
                if df is not None and len(df) >= 60:
                    results[t] = df
                    ok += 1
            print(f"OK={ok}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)
    return results


# ----------------------------------------------
# TECHNICAL SIGNALS  (beta screening)
# ----------------------------------------------

def _rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_signals(ticker, df):
    try:
        close  = df["Close"].dropna().astype(float)
        volume = df["Volume"].dropna().astype(float)
        if len(close) < 60:
            return None

        price = float(close.iloc[-1])
        if price < MIN_PRICE:
            return None

        avg_vol = float(volume.rolling(50).mean().iloc[-1])
        if avg_vol < MIN_AVG_VOLUME:
            return None

        # 20-day mean reversion
        ma20          = float(close.rolling(20).mean().iloc[-1])
        mean_rev      = (price - ma20) / ma20  # negative → oversold

        # RSI-14
        rsi_val       = float(_rsi(close).iloc[-1])

        # Bollinger Bands (20, 2σ)
        std20         = float(close.rolling(20).std().iloc[-1])
        lower_bb      = ma20 - 2 * std20
        upper_bb      = ma20 + 2 * std20
        bb_range      = upper_bb - lower_bb
        bb_pos        = (price - lower_bb) / bb_range if bb_range > 0 else 0.5

        # Volume spike
        recent_vol    = float(volume.iloc[-5:].mean())
        vol_ratio     = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # Momentum
        mom_1m        = float(close.pct_change(21).iloc[-1]) if len(close) >= 21 else 0.0
        mom_3m        = float(close.pct_change(63).iloc[-1]) if len(close) >= 63 else 0.0

        # Composite statistical-arb score (higher = stronger mean-reversion opportunity)
        rsi_score     = max(0.0, (35 - rsi_val) / 35)         # peaks when RSI < 35
        bb_score      = max(0.0, 0.25 - bb_pos) / 0.25        # peaks when near lower band
        mr_score      = max(0.0, -mean_rev * 8)               # peaks when price well below MA
        vol_score     = min(1.0, max(0.0, (vol_ratio - 1) / 3))  # rising volume = confirmation
        stat_arb      = rsi_score * 0.40 + bb_score * 0.30 + mr_score * 0.20 + vol_score * 0.10

        return {
            "ticker":            ticker,
            "current_price":     round(price, 4),
            "rsi":               round(rsi_val, 1),
            "mean_reversion":    round(mean_rev, 5),
            "bb_position":       round(bb_pos, 4),
            "vol_ratio":         round(vol_ratio, 3),
            "mom_1m":            round(mom_1m, 4),
            "mom_3m":            round(mom_3m, 4),
            "stat_arb_score":    round(stat_arb, 5),
        }
    except Exception:
        return None


# ----------------------------------------------
# FUNDAMENTAL SCREENING  (alpha picks)
# ----------------------------------------------

def _fetch_info(ticker):
    try:
        full = yf.Ticker(ticker).info
        return ticker, full
    except Exception:
        return ticker, {}


def get_fundamental_data(tickers, max_workers=12):
    print(f"  Fetching fundamentals for {len(tickers)} tickers (threaded)…")
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_info, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            ticker, info = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(tickers)} done…")
            if not info:
                continue
            try:
                market_cap     = info.get("marketCap") or 0
                if market_cap < 100_000_000:  # skip nano-caps for alpha
                    continue
                pe             = info.get("trailingPE")
                fcf            = info.get("freeCashflow")
                rev_growth     = info.get("revenueGrowth")
                total_debt     = info.get("totalDebt") or 0
                equity         = info.get("totalStockholdersEquity") or 0
                roe            = info.get("returnOnEquity")
                current_price  = info.get("currentPrice") or info.get("regularMarketPrice")

                if not current_price or current_price <= 0:
                    continue

                fcf_yield      = (fcf / market_cap * 100) if fcf and market_cap else None
                de_ratio       = (total_debt / equity) if equity > 0 else None
                roce_proxy     = (roe * 100) if roe else None
                rev_growth_pct = (rev_growth * 100) if rev_growth is not None else None

                rows.append({
                    "ticker":         ticker,
                    "current_price":  current_price,
                    "market_cap":     market_cap,
                    "pe":             pe,
                    "fcf_yield":      fcf_yield,
                    "revenue_growth": rev_growth_pct,
                    "de_ratio":       de_ratio,
                    "roce":           roce_proxy,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def score_alpha(df):
    # yfinance occasionally returns strings (e.g. "Infinity") — coerce all numeric fields
    for col in ("pe", "fcf_yield", "de_ratio", "revenue_growth", "roce"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    s = pd.Series(0.0, index=df.index)
    # Tier 1 — core value (6 pts)
    s += df["pe"].between(0.01, 12).astype(float) * 3
    s += (df["fcf_yield"].fillna(-99) > 5).astype(float) * 3
    # Tier 2 — quality & growth (6 pts)
    s += (df["de_ratio"].fillna(99) < 0.5).astype(float) * 2
    s += (df["revenue_growth"].fillna(-99) > 15).astype(float) * 2
    s += (df["roce"].fillna(-99) > 15).astype(float) * 2
    # Tier 3 — bonus (3 pts)
    s += df["pe"].between(0.01, 8).astype(float) * 1
    s += (df["fcf_yield"].fillna(-99) > 10).astype(float) * 1
    s += (df["revenue_growth"].fillna(-99) > 25).astype(float) * 1
    return s


# ----------------------------------------------
# PORTFOLIO CONSTRUCTION
# ----------------------------------------------

def build_portfolio(beta_signals, alpha_df, sp500_tickers):
    portfolio = []

    # -- BETA PICKS ------------------------------
    beta_df = pd.DataFrame([s for s in beta_signals if s["ticker"] not in sp500_tickers])
    if not beta_df.empty:
        beta_df = beta_df[beta_df["current_price"] >= MIN_PRICE]
        beta_df = beta_df.nlargest(N_BETA_PICKS, "stat_arb_score").reset_index(drop=True)

        n_beta   = len(beta_df)
        beta_per = min(BETA_WEIGHT / n_beta, 0.02)  # cap each position at 2%

        for _, row in beta_df.iterrows():
            dollar_val = PORTFOLIO_SIZE * beta_per
            shares     = dollar_val / row["current_price"]
            portfolio.append({
                "ticker":              row["ticker"],
                "shares":              round(shares, 4),
                "cost_basis":          round(row["current_price"], 4),
                "weight":              round(beta_per, 6),
                "bucket":             "BETA",
                "stat_arb_score":      round(row["stat_arb_score"], 5),
                "rsi":                 round(row["rsi"], 1),
                "mean_reversion_pct":  round(row["mean_reversion"] * 100, 2),
                "bb_position":         round(row["bb_position"], 4),
                "vol_ratio":           round(row["vol_ratio"], 3),
                "mom_1m_pct":          round(row["mom_1m"] * 100, 2),
                "mom_3m_pct":          round(row["mom_3m"] * 100, 2),
            })

    # -- ALPHA PICKS -----------------------------
    if not alpha_df.empty:
        alpha_top  = alpha_df.nlargest(N_ALPHA_PICKS, "score").reset_index(drop=True)
        n_alpha    = len(alpha_top)
        alpha_per  = min(ALPHA_WEIGHT / n_alpha, 0.01)  # cap each at 1%

        for _, row in alpha_top.iterrows():
            if pd.isna(row["current_price"]) or row["current_price"] <= 0:
                continue
            dollar_val = PORTFOLIO_SIZE * alpha_per
            shares     = dollar_val / row["current_price"]
            portfolio.append({
                "ticker":              row["ticker"],
                "shares":              round(shares, 4),
                "cost_basis":          round(row["current_price"], 4),
                "weight":              round(alpha_per, 6),
                "bucket":             "ALPHA",
                "stat_arb_score":      0.0,
                "rsi":                 50.0,
                "mean_reversion_pct":  0.0,
                "bb_position":         0.5,
                "vol_ratio":           1.0,
                "mom_1m_pct":          0.0,
                "mom_3m_pct":          0.0,
            })

    return pd.DataFrame(portfolio)


# ----------------------------------------------
# INVESTMENT THESIS PRINTER
# ----------------------------------------------

def print_thesis(portfolio_df, alpha_df):
    div = "=" * 72
    print(f"\n{div}")
    print("  INVESTMENT THESIS  —  MEDALLION-STYLE 95 / 5 PORTFOLIO")
    print(div)

    # -- BETA ------------------------------------
    print("\n  [ BETA PLAYS: Statistical Edge — 95% Weight ]\n")
    print("  Targeting mean reversion, RSI oversold, Bollinger bounces in")
    print("  non-S&P 500 stocks (less analyst coverage → more mispricing).\n")
    beta_picks = portfolio_df[portfolio_df["bucket"] == "BETA"]
    print(f"  {'TICKER':<8} {'WEIGHT':>7}  {'SCORE':>6}  {'RSI':>5}  {'MR%':>7}  THESIS")
    print(f"  {'-'*8} {'-'*7}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*30}")
    for _, r in beta_picks.iterrows():
        reasons = []
        if r["rsi"] < 30:
            reasons.append(f"RSI={r['rsi']:.0f} OVERSOLD")
        if r["mean_reversion_pct"] < -5:
            reasons.append(f"{r['mean_reversion_pct']:.1f}% below MA20")
        if r["bb_position"] < 0.15:
            reasons.append("near Bollinger lower band")
        if r["vol_ratio"] > 1.5:
            reasons.append(f"vol spike {r['vol_ratio']:.1f}x")
        if not reasons:
            reasons.append(f"stat-arb={r['stat_arb_score']:.4f}")
        thesis = ", ".join(reasons)
        print(f"  {r['ticker']:<8} {r['weight']*100:>6.2f}%  {r['stat_arb_score']:>6.4f}  "
              f"{r['rsi']:>5.1f}  {r['mean_reversion_pct']:>6.1f}%  {thesis}")

    # -- ALPHA -----------------------------------
    print(f"\n  [ ALPHA PLAYS: Multibagger Conviction — 5% Weight ]\n")
    print("  Deep-value + high-growth stocks with strong fundamental moats.\n")
    alpha_picks = portfolio_df[portfolio_df["bucket"] == "ALPHA"]
    for _, r in alpha_picks.iterrows():
        if alpha_df.empty:
            break
        row = alpha_df[alpha_df["ticker"] == r["ticker"]]
        if row.empty:
            continue
        row = row.iloc[0]
        parts = []
        if pd.notna(row.get("pe")) and row["pe"]:
            parts.append(f"P/E={row['pe']:.1f}x")
        if pd.notna(row.get("fcf_yield")) and row["fcf_yield"]:
            parts.append(f"FCF Yield={row['fcf_yield']:.1f}%")
        if pd.notna(row.get("revenue_growth")) and row["revenue_growth"]:
            parts.append(f"RevGrowth={row['revenue_growth']:.0f}%")
        if pd.notna(row.get("de_ratio")) and row["de_ratio"] is not None:
            parts.append(f"D/E={row['de_ratio']:.2f}")
        if pd.notna(row.get("roce")) and row["roce"]:
            parts.append(f"ROCE={row['roce']:.0f}%")
        metrics = " | ".join(parts) if parts else "fundamentals passed screen"
        print(f"  {r['ticker']:<8} {r['weight']*100:.2f}%  score={row.get('score', 0):.0f}  {metrics}")

    print(f"\n{div}\n")


# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MEDALLION-STYLE MARKET SCREENER")
    print(f"  Portfolio: ${PORTFOLIO_SIZE:>12,.0f}")
    print(f"  Beta bucket:  {BETA_WEIGHT*100:.0f}%  (non-S&P500, top {N_BETA_PICKS} stat-arb picks)")
    print(f"  Alpha bucket:  {ALPHA_WEIGHT*100:.0f}%  (multibagger fundamentals, top {N_ALPHA_PICKS})")
    print("=" * 60)

    # -- Step 1: Build universe ------------------
    all_tickers, sp500, indexed = build_universe()
    non_sp500 = all_tickers - sp500
    print(f"\n  Beta universe (non-S&P500): {len(non_sp500):,} tickers")
    print(f"  Alpha universe (S&P500):    {len(sp500):,} tickers\n")

    if len(all_tickers) < 200:
        print("ERROR: Universe too small (<200 tickers). Check network/Wikipedia access.")
        sys.exit(1)

    # -- Step 2: Download prices for beta universe --
    print("=== DOWNLOADING BETA UNIVERSE PRICES ===")
    beta_price_data = download_price_data(non_sp500, period="1y")
    print(f"  Price data for {len(beta_price_data):,} tickers\n")

    # -- Step 3: Calculate technical signals ----
    print("=== CALCULATING TECHNICAL SIGNALS ===")
    beta_signals = []
    for ticker, df in beta_price_data.items():
        sig = calc_signals(ticker, df)
        if sig:
            beta_signals.append(sig)
    print(f"  Valid signals: {len(beta_signals):,} tickers\n")

    if not beta_signals:
        print("ERROR: No valid beta signals. Exiting.")
        sys.exit(1)

    # -- Step 4: Yartseva multibagger scan on indexed universe ----
    # Scope = SP500 + Nasdaq100 + SP400 + SP600 (~1,600). Russell 3000 + Nasdaq Trader
    # microcaps excluded from fundamentals: yfinance rate-limits aggressively above ~2k calls.
    print("=== YARTSEVA MULTIBAGGER SCAN (indexed universe: SP500+N100+SP400+SP600) ===")
    fundamentals_all = get_fundamental_data(list(indexed))
    print(f"  Raw fundamental rows: {len(fundamentals_all)}")

    yartseva_scored = pd.DataFrame()
    if not fundamentals_all.empty:
        fundamentals_all["score"] = score_alpha(fundamentals_all)
        fundamentals_all["in_sp500"] = fundamentals_all["ticker"].isin(sp500)
        yartseva_scored = (
            fundamentals_all[fundamentals_all["score"] >= 5]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        print(f"  Passing threshold (score >= 5): {len(yartseva_scored)}")
        print(f"    - in S&P 500:    {int(yartseva_scored['in_sp500'].sum())}")
        print(f"    - outside SP500: {int((~yartseva_scored['in_sp500']).sum())}\n")
        yartseva_scored.to_csv("yartseva_scan.csv", index=False)
        print("  Saved: yartseva_scan.csv (full-universe multibagger ranking)\n")

    # Alpha portfolio bucket still uses S&P500 subset only (preserves 95/5 design)
    alpha_scored = (
        yartseva_scored[yartseva_scored["in_sp500"]].copy()
        if not yartseva_scored.empty else pd.DataFrame()
    )

    # -- Step 5: Build portfolio -----------------
    print("=== BUILDING MEDALLION PORTFOLIO ===")
    portfolio_df = build_portfolio(beta_signals, alpha_scored, sp500)

    if portfolio_df.empty:
        print("ERROR: Portfolio is empty.")
        sys.exit(1)

    total_weight = portfolio_df["weight"].sum()
    print(f"  Positions:     {len(portfolio_df):>4d}")
    print(f"  Beta picks:    {len(portfolio_df[portfolio_df['bucket']=='BETA']):>4d}")
    print(f"  Alpha picks:   {len(portfolio_df[portfolio_df['bucket']=='ALPHA']):>4d}")
    print(f"  Total weight:  {total_weight*100:.1f}%")

    # -- Step 6: Save CSVs -----------------------
    out_cols = ["ticker", "shares", "cost_basis", "weight"]
    portfolio_df[out_cols].to_csv("portfolio.csv", index=False)
    portfolio_df.to_csv("portfolio_extended.csv", index=False)
    print("\n  Saved: portfolio.csv")
    print("  Saved: portfolio_extended.csv (with all signals)\n")

    # -- Step 7: Print investment thesis ---------
    print_thesis(portfolio_df, alpha_scored)

    print("=== SCREENER COMPLETE ===")
    print("Next: py -3.12 main.py --portfolio portfolio.csv --benchmark SPY\n")


if __name__ == "__main__":
    main()
