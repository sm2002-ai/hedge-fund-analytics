"""S&P 500 constituent universe.

Pulls the current S&P 500 ticker list from Wikipedia. Falls back to an
embedded static snapshot if the fetch fails so the report still runs offline.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

SECTOR_ETFS = {
    "XLK": "Information Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

MACRO_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "Russell 2000",
    "TLT": "20+ Year Treasuries",
    "SHY": "1-3 Year Treasuries",
    "HYG": "High-Yield Credit",
    "LQD": "Investment-Grade Credit",
    "UUP": "US Dollar",
    "USO": "Crude Oil",
    "GLD": "Gold",
    "^VIX": "VIX",
}


# Fallback snapshot — not exhaustive, chosen to cover all 11 GICS sectors so
# the report still has sector breadth if the Wiki fetch fails. Extend or
# replace as needed.
_FALLBACK = [
    # Tech
    ("AAPL", "Information Technology"), ("MSFT", "Information Technology"),
    ("NVDA", "Information Technology"), ("AVGO", "Information Technology"),
    ("ORCL", "Information Technology"), ("CRM", "Information Technology"),
    ("ADBE", "Information Technology"), ("CSCO", "Information Technology"),
    ("AMD", "Information Technology"), ("INTC", "Information Technology"),
    # Communication Services
    ("GOOGL", "Communication Services"), ("META", "Communication Services"),
    ("NFLX", "Communication Services"), ("DIS", "Communication Services"),
    ("TMUS", "Communication Services"), ("VZ", "Communication Services"),
    # Consumer Discretionary
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"), ("MCD", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("SBUX", "Consumer Discretionary"), ("BKNG", "Consumer Discretionary"),
    # Consumer Staples
    ("WMT", "Consumer Staples"), ("PG", "Consumer Staples"),
    ("COST", "Consumer Staples"), ("KO", "Consumer Staples"),
    ("PEP", "Consumer Staples"), ("PM", "Consumer Staples"),
    # Healthcare
    ("UNH", "Healthcare"), ("JNJ", "Healthcare"), ("LLY", "Healthcare"),
    ("ABBV", "Healthcare"), ("MRK", "Healthcare"), ("PFE", "Healthcare"),
    ("TMO", "Healthcare"), ("ABT", "Healthcare"),
    # Financials
    ("BRK-B", "Financials"), ("JPM", "Financials"), ("BAC", "Financials"),
    ("WFC", "Financials"), ("GS", "Financials"), ("MS", "Financials"),
    ("V", "Financials"), ("MA", "Financials"), ("AXP", "Financials"),
    # Industrials
    ("BA", "Industrials"), ("CAT", "Industrials"), ("GE", "Industrials"),
    ("HON", "Industrials"), ("UNP", "Industrials"), ("RTX", "Industrials"),
    ("LMT", "Industrials"), ("DE", "Industrials"),
    # Energy
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
    ("SLB", "Energy"), ("EOG", "Energy"),
    # Utilities
    ("NEE", "Utilities"), ("SO", "Utilities"), ("DUK", "Utilities"),
    # Materials
    ("LIN", "Materials"), ("SHW", "Materials"), ("APD", "Materials"),
    # Real Estate
    ("PLD", "Real Estate"), ("AMT", "Real Estate"), ("EQIX", "Real Estate"),
]


def fetch_sp500_universe() -> pd.DataFrame:
    """Return a DataFrame with columns [ticker, sector] for the S&P 500.

    Fetches the live list from Wikipedia; falls back to a static snapshot
    on any error so the report still runs offline / in CI.
    """
    try:
        resp = requests.get(
            WIKI_URL,
            headers={"User-Agent": "Mozilla/5.0 (hedge-fund-analytics)"},
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = tables[0]
        # Wikipedia column names drift — normalize to lowercase + strip.
        df.columns = [c.strip().lower() for c in df.columns]
        # yfinance uses dash instead of dot for class shares (BRK.B -> BRK-B).
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
        sector_col = "gics sector" if "gics sector" in df.columns else "sector"
        out = df[["symbol", sector_col]].rename(
            columns={"symbol": "ticker", sector_col: "sector"}
        )
        return out.reset_index(drop=True)
    except Exception as e:
        print(f"[market_universe] Wiki fetch failed ({e}); using fallback list.")
        return pd.DataFrame(_FALLBACK, columns=["ticker", "sector"])


def filter_universe(
    universe: pd.DataFrame, sectors: Iterable[str] | None = None
) -> pd.DataFrame:
    if sectors is None:
        return universe
    wanted = set(sectors)
    return universe[universe["sector"].isin(wanted)].reset_index(drop=True)
