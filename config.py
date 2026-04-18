import os
from dotenv import load_dotenv

load_dotenv()

PORTFOLIO_TICKERS = ["TENB", "VRNS", "LOAR", "NOW", "PANW", "CPNG", "WVE"]
BENCHMARK = "SPY"
RISK_FREE_RATE = 0.0525  # 5.25% annualized (current Fed funds rate)
CONFIDENCE_LEVELS = [0.95, 0.99]
LOOKBACK_DAYS = 252
MONTE_CARLO_SIMS = 10000
MONTE_CARLO_HORIZON = 252

SECTOR_MAP = {
    "TENB": "Cybersecurity",
    "VRNS": "Cybersecurity",
    "PANW": "Cybersecurity",
    "LOAR": "Aerospace & Defense",
    "NOW": "Software & Cloud",
    "CPNG": "E-Commerce",
    "WVE": "Healthcare & Biotech",
}

REPORT_TITLE = os.getenv("REPORT_TITLE", "Hedge Fund Analytics Report")
FUND_NAME = os.getenv("FUND_NAME", "Alpha Strategy Fund")

EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Scoring methodology weights used by financial_analysis.py
ANALYSIS_SCORE_WEIGHTS = {
    "profitability": 0.35,   # gross/op/net margin, ROE, ROA
    "value":         0.25,   # P/E, fwd P/E, P/B, EV/EBITDA (positive values only)
    "growth":        0.25,   # revenue YoY growth
    "risk":          0.15,   # beta, debt/equity (lower = better)
}
PORTFOLIO_LAST_UPDATED = "2026-04-18"

BLOOMBERG_NAVY = "#003153"
BLOOMBERG_GOLD = "#F4C430"
BLOOMBERG_SLATE = "#708090"
BLOOMBERG_LIGHT = "#F5F5F0"
CHART_PALETTE = ["#003153", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]
