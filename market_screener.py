"""
market_screener.py — Full US market screener for alpha/beta opportunity discovery.

Downloads S&P 500 + NASDAQ 100 universe (~600 unique tickers), computes alpha,
beta, Sharpe, momentum, and volume trend for each, then builds a top-15 portfolio.
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import io

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from config import BENCHMARK, RISK_FREE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Universe construction
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _safe_read_html(url: str, table_index: int, column: str) -> list[str]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[table_index]
        if column not in df.columns:
            for c in df.columns:
                if str(c).lower() == column.lower():
                    column = c
                    break
        if column not in df.columns:
            return []
        return df[column].dropna().tolist()
    except Exception as e:
        log.warning(f"Failed to fetch {url} (table={table_index}, col={column}): {e}")
        return []


# Hardcoded fallback — curated S&P 500 + NASDAQ 100 constituents
_FALLBACK_TICKERS = """
AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA BRK-B JPM
UNH LLY XOM JNJ V AVGO MA PG HD CVX
MRK ABBV KO COST PEP ADBE WMT AMD NFLX CRM
BAC TMO CSCO PFE DIS ABT WFC ACN TXN LIN
DHR PM ORCL MCD INTU QCOM UPS IBM GE RTX
CAT SPGI MS HON GS AMGN ISRG BKNG SYK MDLZ
BLK GILD T VRTX AXP LOW CI ADI REGN MMC
ZTS PLD SCHW CB DE FI PANW SNPS CDNS KLAC
AMT LRCX ANET MRVL DXCM ON SMCI FTNT CRWD MU
TEAM TTD SNOW DDOG ZS OKTA HUBS NET VEEV PAYC
BILL SQ TWLO SHOP RBLX DOCU FROG ESTC SMAR ASAN
CFLT NTNX MDB GTLB BRZE ZI TOST APP DUOL SOUN
MSTR COIN HOOD SOFI UPST AFRM LMND OPEN OPENDOOR
AXON TMDX KRYS RXRX HALO ARGX ALNY INCY BMRN IONS
EXEL PCVX FATE NTLA CRSP EDIT BEAM NTLA SANA GRPH
APLS ARWR SPRING PRAXIS KYMR NUVL AUPH ACCD RLAY PTGX
GNRC PODD HOLX TNDM IRTC NVCR AXNX PNTM RXDX NNOX
PAYC NOW WDAY VEEV COUP SPSC APPF PCTY SMAR LPSN
ROKU FUBO PARA WBD DIS NFLX SPOT BMBL MTCH SNAP
PINS TWTR ZG EXPI OPEN RDFN TREX UFP BECN IBP
RH WSM TSCO FIVE OLLI BJ COST TGT WMT TJX
ULTA LULU NKE ONON CROX SKX GPS URBN FIGS LOVE
CHWY PETQ PET BARK ZGNX KCGI MNST CELH VITL FRPT
USFD SYY KR PFGC CHEF JBSS POST GIS CPB MKC
ADM BG CTVA FMC MOS NTR CF DOW LYB CE
EMN HUN WLK AXTA RPM SHW PPG FUL H IFF
APD LIN ECL ALB LTHM LIVENT SQM CIEN IIPR GLBE
KSPI NU StoneCo MELI GRAB SEA BIDU JD PDD BABA
TCEHY NTES YY QFIN TIGR FUTU LI NIO XPEV RIVN
LCID FSR GOEV WKHS RIDE SOLO KNDI NKLA HYLN AYRO
SPCE ACIC BFLY OUST LIDR AEVA MVIS LAZR INVZ OPAL
ARRY ENPH SEDG FSLR RUN NOVA CWEN NEE AES ETR
XEL DUK SO D AEE LNT NI WEC ES CNP
AWK SRE PCG EIX PPL AEP FE EXC PEG ED
OKE WMB KMI EPD ET MPLX PAA TRGP ENB TRP
PSX VLO MPC HES DVN FANG PXD OXY COP EOG
EQT AR SWN RRC CHK CNX AR CTRA ESTE PR
AM SMLP HESM CEQP NBLX MMLP BSM DKL PBFX PBF
VLO DINO HFC CLMT HFC PARR CALUMET NTUS HDSN CLNE
AMR ARCH BTU CEIX FANG VNOM MPLX HESM SMLP CPLP
""".split()

_FALLBACK_TICKERS = [
    t.strip().upper() for t in _FALLBACK_TICKERS
    if t.strip() and len(t.strip()) <= 6 and "." not in t and "/" not in t
]
# Remove obvious duplicates
_FALLBACK_TICKERS = list(dict.fromkeys(_FALLBACK_TICKERS))


def get_universe() -> list[str]:
    """Return deduplicated list of US equity tickers."""
    tickers: list[str] = []

    log.info("Fetching S&P 500 list from Wikipedia …")
    sp500 = _safe_read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol"
    )
    log.info(f"  S&P 500: {len(sp500)} tickers")
    tickers.extend(sp500)

    log.info("Fetching NASDAQ-100 list from Wikipedia …")
    ndx: list[str] = []
    for idx in range(8):
        ndx = _safe_read_html("https://en.wikipedia.org/wiki/Nasdaq-100", idx, "Ticker")
        if len(ndx) >= 90:
            break
        ndx = _safe_read_html("https://en.wikipedia.org/wiki/Nasdaq-100", idx, "Symbol")
        if len(ndx) >= 90:
            break
    log.info(f"  NASDAQ-100: {len(ndx)} tickers")
    tickers.extend(ndx)

    if len(tickers) < 50:
        log.warning("Wikipedia scraping yielded too few tickers — using hardcoded fallback universe.")
        tickers = _FALLBACK_TICKERS.copy()
    else:
        # Merge with fallback to pad out the universe
        tickers.extend(_FALLBACK_TICKERS)

    # Clean: strip whitespace, drop empty, uppercase, remove non-equity formats
    cleaned: list[str] = []
    for t in tickers:
        t = str(t).strip().upper()
        if not t or len(t) > 6 or " " in t or "." in t:
            continue
        cleaned.append(t)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    log.info(f"Total universe: {len(unique)} unique tickers")
    return unique


# ---------------------------------------------------------------------------
# 2. Batch price download
# ---------------------------------------------------------------------------

CHUNK_SIZE = 100


def download_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Batch-download adjusted closing prices in chunks to avoid timeouts.
    Returns a DataFrame with tickers as columns and dates as index.
    """
    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []

    chunks = [tickers[i : i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    log.info(f"Downloading prices for {len(tickers)} tickers in {len(chunks)} chunks …")

    for i, chunk in enumerate(chunks):
        log.info(f"  Chunk {i+1}/{len(chunks)} ({len(chunk)} tickers) …")
        try:
            raw = yf.download(
                chunk,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yfinance returns MultiIndex columns when multiple tickers
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
            else:
                # Single ticker edge case
                close = raw[["Close"]]
                close.columns = chunk[:1]

            # Drop columns that are all-NaN
            close = close.dropna(axis=1, how="all")
            all_frames.append(close)
        except Exception as e:
            log.warning(f"  Chunk {i+1} failed: {e}")
            failed.extend(chunk)

    if not all_frames:
        raise RuntimeError("All download chunks failed — check internet connection.")

    prices = pd.concat(all_frames, axis=1)
    # Drop duplicate columns (same ticker downloaded twice)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    log.info(f"Downloaded prices for {prices.shape[1]} tickers ({len(failed)} failed chunks)")
    return prices


# ---------------------------------------------------------------------------
# 3. Metric calculations (vectorized)
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
RISK_FREE_DAILY = RISK_FREE_RATE / TRADING_DAYS


def compute_metrics(prices: pd.DataFrame, benchmark_prices: pd.Series) -> pd.DataFrame:
    """
    Compute all screening metrics for every ticker in prices.
    Returns a DataFrame with one row per ticker.
    """
    # Daily returns
    rets = prices.pct_change().iloc[1:]
    bench_rets = benchmark_prices.pct_change().iloc[1:]

    # Align on common dates
    common_idx = rets.index.intersection(bench_rets.index)
    rets = rets.loc[common_idx]
    bench = bench_rets.loc[common_idx]

    n = len(rets)
    if n < 60:
        raise ValueError("Less than 60 trading days of data — cannot compute metrics.")

    bench_arr = bench.values  # shape (n,)

    results: list[dict] = []

    for ticker in rets.columns:
        r = rets[ticker].dropna()
        if len(r) < 60:
            continue

        r_arr = r.values
        r_mean = r_arr.mean()

        # ---------- Beta / Alpha ----------
        # Align r_arr with bench_arr on same dates
        common = r.index.intersection(bench.index)
        if len(common) < 60:
            continue
        r_c = r.loc[common].values
        b_c = bench.loc[common].values
        cov = np.cov(r_c, b_c, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
        # Jensen's alpha (annualized) — use same-period bench mean so sparse tickers aren't biased
        b_c_mean = b_c.mean()
        alpha = (r_mean - RISK_FREE_DAILY - beta * (b_c_mean - RISK_FREE_DAILY)) * TRADING_DAYS

        # ---------- Sharpe ----------
        excess = r_arr - RISK_FREE_DAILY
        vol = r_arr.std()
        sharpe = (excess.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan

        # ---------- Annualized volatility ----------
        ann_vol = vol * np.sqrt(TRADING_DAYS)

        # ---------- Returns ----------
        def _period_return(days: int) -> float:
            # Use available data up to `days`; require at least half the window
            available = len(r_arr)
            if available < max(days // 2, 20):
                return np.nan
            return float(np.prod(1 + r_arr[-min(days, available):]) - 1)

        ret_1m = _period_return(21)
        ret_3m = _period_return(63)
        ret_6m = _period_return(126)
        ret_1y = _period_return(252)

        # ---------- Momentum (price vs MAs) ----------
        p = prices[ticker].dropna()
        ma50_score = np.nan
        ma200_score = np.nan
        if len(p) >= 50:
            ma50 = p.rolling(50).mean().iloc[-1]
            ma50_score = float((p.iloc[-1] - ma50) / ma50) if ma50 != 0 else np.nan
        if len(p) >= 200:
            ma200 = p.rolling(200).mean().iloc[-1]
            ma200_score = float((p.iloc[-1] - ma200) / ma200) if ma200 != 0 else np.nan

        momentum = np.nanmean([ma50_score, ma200_score]) if not (np.isnan(ma50_score) and np.isnan(ma200_score)) else np.nan


        results.append(
            {
                "ticker": ticker,
                "alpha": alpha,
                "beta": beta,
                "sharpe": sharpe,
                "ann_vol": ann_vol,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "ret_1y": ret_1y,
                "momentum": momentum,
                "ma50_score": ma50_score,
                "ma200_score": ma200_score,
            }
        )

    df = pd.DataFrame(results).set_index("ticker")

    # ---------- Composite score ----------
    # Normalise each component to [0,1] then weight
    def _rank_norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    alpha_n = _rank_norm(df["alpha"].fillna(df["alpha"].median()))
    sharpe_n = _rank_norm(df["sharpe"].fillna(df["sharpe"].median()))
    mom_n = _rank_norm(df["momentum"].fillna(df["momentum"].median()))
    ret3m_n = _rank_norm(df["ret_3m"].fillna(df["ret_3m"].median()))

    df["composite_score"] = 0.40 * alpha_n + 0.25 * sharpe_n + 0.20 * mom_n + 0.15 * ret3m_n

    return df


# ---------------------------------------------------------------------------
# 4. Volume trend — separate download
# ---------------------------------------------------------------------------

def enrich_volume_trend(metrics: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Download volume data and compute 20d-vs-50d volume trend."""
    log.info("Fetching volume data for volume trend calculation …")
    vol_chunks = [tickers[i : i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    vol_trend_map: dict[str, float] = {}

    for i, chunk in enumerate(vol_chunks):
        try:
            raw = yf.download(chunk, period="3mo", auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                vol_df = raw["Volume"]
            else:
                vol_df = raw[["Volume"]]
                vol_df.columns = chunk[:1]

            for tk in vol_df.columns:
                v = vol_df[tk].dropna()
                if len(v) >= 50:
                    v20 = v.iloc[-20:].mean()
                    v50 = v.iloc[-50:].mean()
                    vol_trend_map[tk] = float(v20 / v50 - 1) if v50 > 0 else 0.0
        except Exception as e:
            log.warning(f"Volume chunk {i+1} failed: {e}")

    vol_series = pd.Series(vol_trend_map, name="vol_trend")
    metrics = metrics.join(vol_series, how="left")

    # Recompute composite with vol_trend
    def _rank_norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    vol_n = _rank_norm(metrics["vol_trend"].fillna(0.0))
    alpha_n = _rank_norm(metrics["alpha"].fillna(metrics["alpha"].median()))
    sharpe_n = _rank_norm(metrics["sharpe"].fillna(metrics["sharpe"].median()))
    mom_n = _rank_norm(metrics["momentum"].fillna(metrics["momentum"].median()))
    ret3m_n = _rank_norm(metrics["ret_3m"].fillna(metrics["ret_3m"].median()))

    metrics["composite_score"] = (
        0.35 * alpha_n
        + 0.25 * sharpe_n
        + 0.20 * mom_n
        + 0.10 * ret3m_n
        + 0.10 * vol_n
    )

    return metrics


# ---------------------------------------------------------------------------
# 5. Portfolio weight optimisation (equal-weight with vol-parity tilt)
# ---------------------------------------------------------------------------

def compute_weights(tickers: list[str], metrics: pd.DataFrame) -> dict[str, float]:
    """
    Inverse-volatility weighting for portfolio construction.
    Fallback to equal-weight if vol data unavailable.
    """
    vols = metrics.loc[tickers, "ann_vol"].replace(0, np.nan).dropna()
    if len(vols) == 0:
        w = {t: round(1.0 / len(tickers), 4) for t in tickers}
        return w
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return {t: round(float(weights.get(t, 1.0 / len(tickers))), 4) for t in tickers}


# ---------------------------------------------------------------------------
# 6. Output helpers
# ---------------------------------------------------------------------------

def save_screen_results(metrics: pd.DataFrame, output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.today().strftime("%Y%m%d")
    path = os.path.join(output_dir, f"market_screen_{date_str}.csv")
    metrics.sort_values("composite_score", ascending=False).to_csv(path)
    log.info(f"Full screening results saved → {path}")
    return path


def update_portfolio_csv(top_picks: list[str], weights: dict[str, float], path: str = "portfolio.csv") -> None:
    rows = []
    for ticker in top_picks:
        rows.append(
            {
                "ticker": ticker,
                "shares": 0,          # placeholder — no cost basis from screener
                "cost_basis": 0.0,
                "sector": "Screened",
                "weight": weights.get(ticker, round(1.0 / len(top_picks), 4)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"portfolio.csv updated with {len(top_picks)} holdings → {path}")


def print_table(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    display = df[cols] if cols else df
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(display.to_string())
    print()


# ---------------------------------------------------------------------------
# 7. Main screening pipeline
# ---------------------------------------------------------------------------

def run_screener(top_n: int = 15) -> pd.DataFrame:
    t_start = datetime.now()

    # -- Universe --
    universe = get_universe()
    if not universe:
        raise RuntimeError("Could not build ticker universe — check network access.")

    # -- Download benchmark separately --
    log.info(f"Downloading benchmark {BENCHMARK} prices …")
    bench_raw = yf.download(BENCHMARK, period="1y", auto_adjust=True, progress=False)
    if isinstance(bench_raw.columns, pd.MultiIndex):
        bench_prices = bench_raw["Close"][BENCHMARK]
    else:
        bench_prices = bench_raw["Close"]
    bench_prices = bench_prices.dropna()

    # -- Download universe prices --
    prices = download_prices(universe, period="1y")

    # Align benchmark to same date range
    common_dates = prices.index.intersection(bench_prices.index)
    prices = prices.loc[common_dates]
    bench_prices = bench_prices.loc[common_dates]

    # Drop tickers with too little data
    min_obs = 180
    prices = prices.loc[:, prices.notna().sum() >= min_obs]
    log.info(f"Tickers with ≥{min_obs} observations: {prices.shape[1]}")

    # -- Compute metrics --
    log.info("Computing alpha / beta / Sharpe / momentum for all tickers …")
    metrics = compute_metrics(prices, bench_prices)
    log.info(f"Metrics computed for {len(metrics)} tickers")

    # -- Enrich with volume trend --
    valid_tickers = metrics.index.tolist()
    metrics = enrich_volume_trend(metrics, valid_tickers)

    # -- Drop any rows with NaN alpha or Sharpe --
    metrics = metrics.dropna(subset=["alpha", "sharpe"])
    log.info(f"Clean metrics rows: {len(metrics)}")

    # -- Save full results --
    screen_path = save_screen_results(metrics)

    # -- Top 20 by Alpha --
    top_alpha = metrics.nlargest(20, "alpha")
    print_table(
        top_alpha,
        "TOP 20 STOCKS BY ALPHA (Jensen's Alpha vs SPY, 1Y)",
        ["alpha", "beta", "sharpe", "ret_1y", "momentum"],
    )

    # -- Top 20 by Composite Score --
    top_composite = metrics.nlargest(20, "composite_score")
    print_table(
        top_composite,
        "TOP 20 STOCKS BY COMPOSITE SCORE (Alpha 35% + Sharpe 25% + Momentum 20% + Return 10% + Volume 10%)",
        ["composite_score", "alpha", "sharpe", "momentum", "ret_3m"],
    )

    # -- Recommended portfolio: top 15 by composite --
    portfolio_picks = top_composite.head(top_n).index.tolist()
    weights = compute_weights(portfolio_picks, metrics)

    print(f"\n{'='*70}")
    print(f"  RECOMMENDED PORTFOLIO — TOP {top_n} PICKS")
    print(f"{'='*70}")
    for ticker in portfolio_picks:
        row = metrics.loc[ticker]
        print(
            f"  {ticker:<8}  weight={weights[ticker]:.2%}  "
            f"alpha={row['alpha']:+.2%}  sharpe={row['sharpe']:.2f}  "
            f"momentum={row.get('momentum', float('nan')):.2%}"
        )

    # -- Update portfolio.csv --
    update_portfolio_csv(portfolio_picks, weights)

    elapsed = (datetime.now() - t_start).total_seconds()
    log.info(f"Screening complete in {elapsed:.1f}s. Results → {screen_path}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = run_screener(top_n=15)
    sys.exit(0)
