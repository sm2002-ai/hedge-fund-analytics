"""Market-wide hedge-fund-style opportunity report.

Fetches S&P 500 prices, sector ETFs, macro indices and Fama-French factors,
then assembles a hedge-fund-style research memo (regime, macro, sectors,
stock screens, top ideas) rendered to HTML and emailed.
"""

from __future__ import annotations

import datetime
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import (
    EMAIL_FROM, EMAIL_TO, FUND_NAME, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER,
)
from factor_analysis import fetch_fama_french_5_factors
from market_analysis import (
    factor_performance, macro_snapshot, market_regime, market_risk_snapshot,
    sector_leaderboard,
)
from market_screener import (
    build_screener_frame, low_vol_high_sharpe, momentum_leaders, top_n, value_bounce,
)
from market_universe import MACRO_TICKERS, SECTOR_ETFS, fetch_sp500_universe


TEMPLATE_FILE = "market_template.html"


def _download_close(tickers: list, start: str, chunk_size: int = 100) -> pd.DataFrame:
    """Download adjusted close prices in chunks (yfinance chokes on 500+ at once)."""
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        raw = yf.download(
            chunk, start=start, progress=False, auto_adjust=True,
            group_by="column", threads=True,
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close = raw["Close"]
            else:
                close = raw.xs("Close", axis=1, level=-1)
        else:
            close = raw["Close"].to_frame(name=chunk[0])
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def generate_market_report(
    lookback_days: int = 252,
    top_ideas: int = 10,
    output_path: str | None = None,
) -> str:
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=int(lookback_days * 2))).strftime("%Y-%m-%d")
    output_path = output_path or f"market_report_{today.isoformat()}.html"

    print("[1/6] Fetching S&P 500 universe...")
    universe = fetch_sp500_universe()
    tickers = universe["ticker"].tolist()
    sector_series = universe.set_index("ticker")["sector"]

    print(f"[2/6] Downloading prices for {len(tickers)} stocks + sector ETFs + macro...")
    extras = list(SECTOR_ETFS.keys()) + list(MACRO_TICKERS.keys())
    all_prices = _download_close(tickers + extras, start=start)
    stock_prices = all_prices.reindex(columns=[t for t in tickers if t in all_prices.columns])
    etf_prices = all_prices.reindex(columns=[t for t in extras if t in all_prices.columns])

    print("[3/6] Computing regime, macro, sector leaderboard...")
    spy = etf_prices["SPY"].dropna() if "SPY" in etf_prices.columns else pd.Series(dtype=float)
    vix = etf_prices["^VIX"].dropna() if "^VIX" in etf_prices.columns else None
    regime = market_regime(spy, vix) if not spy.empty else {}
    macro = macro_snapshot(etf_prices, MACRO_TICKERS)
    sector_board = sector_leaderboard(etf_prices, SECTOR_ETFS) if "SPY" in etf_prices.columns else []
    spy_rets = spy.pct_change().dropna()
    risk_snap = market_risk_snapshot(spy_rets) if not spy_rets.empty else {}

    print("[4/6] Fetching Fama-French factors...")
    try:
        ff5 = fetch_fama_french_5_factors(
            start_date=start, end_date=today.isoformat(),
        )
        factor_perf = factor_performance(ff5)
    except Exception as e:
        print(f"  Factor fetch failed: {e}")
        factor_perf = []

    print(f"[5/6] Running screens on {stock_prices.shape[1]} tickers...")
    screener = build_screener_frame(stock_prices, sectors=sector_series)
    ideas_df = top_n(screener, n=top_ideas, prices=stock_prices)
    ideas = _rows(ideas_df)
    momo = _rows(momentum_leaders(screener, 10))
    quality = _rows(low_vol_high_sharpe(screener, 10))
    bounce = _rows(value_bounce(screener, 10))

    print("[6/6] Rendering HTML...")
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(TEMPLATE_FILE)
    html = template.render(
        fund_name=FUND_NAME,
        report_date=today.strftime("%B %d, %Y"),
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        lookback_days=lookback_days,
        universe_size=stock_prices.shape[1],
        regime=regime,
        macro=macro,
        sectors=sector_board,
        factors=factor_perf,
        risk=risk_snap,
        top_ideas=ideas,
        momentum_leaders=momo,
        quality_leaders=quality,
        value_bounce=bounce,
    )
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Report written → {output_path}")
    return html


def _rows(df: pd.DataFrame) -> list[dict]:
    """Convert a screener DataFrame to a template-friendly list of dicts."""
    out = []
    for ticker, row in df.iterrows():
        out.append({
            "ticker": ticker,
            "sector": row.get("sector", ""),
            "momentum": float(row.get("momentum", 0)),
            "sharpe": float(row.get("sharpe", 0)),
            "volatility": float(row.get("volatility", 0)),
            "max_drawdown": float(row.get("max_drawdown", 0)),
            "trend": float(row.get("trend", 0)),
            "composite": float(row.get("composite", 0)),
        })
    return out


def send_market_report(html: str, attachment_path: str | None = None) -> None:
    if not EMAIL_TO or not EMAIL_FROM:
        print("[market_report] Email not configured — skipping send.")
        return

    today = datetime.date.today().isoformat()
    subject = f"{FUND_NAME} — Market Opportunities Report {today}"
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html", "utf-8"))

    if attachment_path and Path(attachment_path).exists():
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="html")
        part.add_header(
            "Content-Disposition", "attachment", filename=os.path.basename(attachment_path)
        )
        msg.attach(part)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    print(f"[market_report] Emailed market report to {EMAIL_TO}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate market-wide opportunities report.")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--top-ideas", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()

    output_path = args.output or f"market_report_{datetime.date.today().isoformat()}.html"
    html = generate_market_report(
        lookback_days=args.lookback,
        top_ideas=args.top_ideas,
        output_path=output_path,
    )
    if not args.no_email:
        try:
            send_market_report(html, attachment_path=output_path)
        except Exception as e:
            print(f"[market_report] Email delivery failed: {e}")


if __name__ == "__main__":
    main()
