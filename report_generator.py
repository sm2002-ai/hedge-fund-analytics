import os
import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
import yfinance as yf
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import (
    PORTFOLIO_TICKERS, BENCHMARK, SECTOR_MAP, FUND_NAME, REPORT_TITLE,
    EMAIL_TO, EMAIL_FROM, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
)
from risk_metrics import compute_all_risk_metrics
from factor_analysis import fetch_fama_french_5_factors, run_ff5_regression
from stress_testing import run_all_stress_tests
from attribution import sector_attribution
from chart_generator import generate_all_charts


def fetch_price_data(tickers: list, benchmark: str, lookback_days: int = 252) -> tuple:
    """Download adjusted close prices for portfolio and benchmark."""
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
    all_tickers = tickers + [benchmark]
    raw = yf.download(all_tickers, start=start, progress=False, auto_adjust=True)["Close"]

    if len(all_tickers) == 1:
        raw = raw.to_frame(name=all_tickers[0])

    raw = raw.dropna(how="all").tail(lookback_days + 1)
    returns = raw.pct_change().dropna()
    return raw, returns


def build_portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted portfolio daily returns."""
    port_tickers = [t for t in weights if t in returns.columns]
    w = pd.Series({t: weights[t] for t in port_tickers})
    w = w / w.sum()
    return (returns[port_tickers] * w).sum(axis=1).rename("Portfolio")


def build_holdings_table(portfolio_df: pd.DataFrame, prices: pd.DataFrame) -> list:
    """Build holdings detail list with current prices and P&L."""
    holdings = []
    total_value = 0.0
    rows = []
    for _, row in portfolio_df.iterrows():
        ticker = row["ticker"]
        shares = float(row["shares"])
        cost = float(row["cost_basis"])
        sector = row.get("sector", SECTOR_MAP.get(ticker, "Other"))
        if ticker in prices.columns:
            price = float(prices[ticker].iloc[-1])
        else:
            price = cost
        value = shares * price
        pnl = value - shares * cost
        pnl_pct = (price / cost - 1) * 100
        total_value += value
        rows.append({
            "ticker": ticker, "sector": sector, "shares": int(shares),
            "cost_basis": cost, "price": price, "value": value,
            "pnl": pnl, "pnl_pct": pnl_pct, "weight": 0.0,
        })

    for r in rows:
        r["weight"] = r["value"] / total_value if total_value > 0 else 0.0
    return rows, total_value


def generate_report(portfolio_df: pd.DataFrame, lookback_days: int = 252,
                    output_path: str = "report.html") -> str:
    """Full pipeline: fetch data → compute metrics → charts → render HTML."""
    tickers = portfolio_df["ticker"].tolist()
    weights_raw = {r["ticker"]: float(r["weight"]) for _, r in portfolio_df.iterrows()}

    print("[1/7] Fetching price data...")
    prices, returns = fetch_price_data(tickers, BENCHMARK, lookback_days)
    port_returns = build_portfolio_returns(returns, weights_raw)
    bench_returns = returns[BENCHMARK].rename("Benchmark")

    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    port_returns = aligned["Portfolio"]
    bench_returns = aligned["Benchmark"]

    print("[2/7] Computing risk metrics...")
    metrics_dict = compute_all_risk_metrics(port_returns, bench_returns)

    class MetricsObj:
        def __init__(self, d):
            self.__dict__.update(d)
            if isinstance(d.get("win_loss"), dict):
                self.win_loss = type("WL", (), d["win_loss"])()

    metrics = MetricsObj(metrics_dict)

    print("[3/7] Running factor analysis...")
    try:
        start_str = port_returns.index[0].strftime("%Y-%m-%d")
        end_str = port_returns.index[-1].strftime("%Y-%m-%d")
        ff5 = fetch_fama_french_5_factors(start_str, end_str)
        factor_data = run_ff5_regression(port_returns, ff5)
    except Exception as e:
        print(f"  Factor analysis failed: {e}")
        factor_data = None

    print("[4/7] Running stress tests...")
    try:
        stress_data = run_all_stress_tests(port_returns, bench_returns, weights_raw, metrics.beta)
        class StressObj:
            def __init__(self, d):
                self.__dict__.update(d)
                for k in ["rate_hike_100bps", "rate_hike_200bps", "recession_mild", "recession_severe"]:
                    if k in d and isinstance(d[k], dict):
                        setattr(self, k, type("SO", (), d[k])())
        stress_obj = StressObj(stress_data)
    except Exception as e:
        print(f"  Stress tests failed: {e}")
        stress_obj = None

    print("[5/7] Computing attribution...")
    sector_weights = {}
    sector_returns_map = {}
    for _, row in portfolio_df.iterrows():
        t = row["ticker"]
        sector = SECTOR_MAP.get(t, "Other")
        w = weights_raw.get(t, 0)
        if t in returns.columns:
            r = float((1 + returns[t]).prod() - 1)
        else:
            r = 0.0
        sector_weights[sector] = sector_weights.get(sector, 0) + w
        sector_returns_map[sector] = sector_returns_map.get(sector, 0) + r * w

    for s in sector_returns_map:
        if sector_weights[s] > 0:
            sector_returns_map[s] /= sector_weights[s]

    try:
        attr_data = sector_attribution(weights_raw, {t: float((1 + returns[t]).prod() - 1)
                                                      if t in returns.columns else 0.0
                                                      for t in tickers})
        class AttrObj:
            def __init__(self, d):
                self.__dict__.update(d)
        attr_obj = AttrObj(attr_data)
    except Exception as e:
        print(f"  Attribution failed: {e}")
        attr_obj = None

    print("[6/7] Generating charts...")
    charts = generate_all_charts(
        port_returns, bench_returns, metrics_dict, factor_data,
        stress_data if stress_obj else None, sector_weights,
    )

    print("[7/7] Rendering HTML report...")
    holdings, nav = build_holdings_table(portfolio_df, prices)

    bench_annual = float((1 + bench_returns.mean()) ** 252 - 1)
    active_return = metrics.annualized_return - bench_annual

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("template.html")
    html = template.render(
        fund_name=FUND_NAME,
        report_title=REPORT_TITLE,
        report_date=datetime.date.today().strftime("%B %d, %Y"),
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        benchmark_label=BENCHMARK,
        lookback_days=lookback_days,
        nav=nav,
        n_holdings=len(holdings),
        total_return=metrics.total_return,
        active_return=active_return,
        bench_annual=bench_annual,
        metrics=metrics,
        factor_data=factor_data,
        stress_data=stress_obj,
        attribution=attr_obj,
        holdings=holdings,
        charts=charts,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written → {output_path}")
    return html


def send_email_report(html: str, subject: str = None, attachment_path: str = None):
    """Send HTML report via SMTP."""
    if not EMAIL_TO or not EMAIL_FROM:
        print("Email not configured — skipping send.")
        return

    subject = subject or f"{FUND_NAME} — Daily Report {datetime.date.today()}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html", "utf-8"))

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
        msg.attach(part)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    print(f"Report emailed to {EMAIL_TO}")
