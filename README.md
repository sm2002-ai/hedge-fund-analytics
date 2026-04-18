# Hedge Fund Analytics

Institutional-grade portfolio analytics and daily reporting engine. Generates Bloomberg/Goldman-style HTML reports with embedded charts, delivered via email on a daily schedule.

## Features

- **Risk Metrics** — VaR (parametric + historical, 95%/99%), CVaR/Expected Shortfall, rolling VaR (30/60/90-day), Sharpe, Sortino, Calmar, Information Ratio, Max Drawdown with recovery analysis, Beta, Alpha, Tracking Error, Win/Loss ratio
- **Factor Analysis** — Fama-French 5-factor regression (Kenneth French Data Library), factor loadings, alpha, R², factor return attribution, rolling exposures
- **Stress Testing** — Historical crisis replay (GFC 2008, COVID 2020, Dotcom, 2022 Rate Shock), single-day market shocks (−5% to −20%), rate hike scenarios (+100/200bps), recession scenarios (mild/severe)
- **Performance Attribution** — Brinson-Fachler sector attribution: allocation effect, selection effect, interaction effect
- **Institutional Charts** — Bloomberg/Goldman style matplotlib charts: cumulative performance, drawdown, VaR distribution, factor heatmap, sector allocation, stress test waterfall, Monte Carlo fan chart (all embedded as base64 PNG)
- **Email Delivery** — Automated daily report at 7:00 AM Guatemala time via GitHub Actions

## Setup

### 1. Clone and install

```bash
git clone https://github.com/sm2002-ai/hedge-fund-analytics.git
cd hedge-fund-analytics
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your SMTP credentials
```

### 3. Customize portfolio

Edit `portfolio.csv` with your holdings:

```csv
ticker,shares,cost_basis,sector,weight
TENB,30,16.9833,Cybersecurity,0.275
VRNS,16,22.2169,Cybersecurity,0.184
```

### 4. Run locally

```bash
# Generate report (no email)
python main.py --no-email

# Full run with email
python main.py --portfolio portfolio.csv --email you@example.com

# Custom lookback period
python main.py --lookback 126 --output q2_report.html --no-email
```

## GitHub Actions (Automated Daily Reports)

Add these secrets to your repository (`Settings → Secrets → Actions`):

| Secret | Description |
|--------|-------------|
| `EMAIL_TO` | Report recipient email |
| `EMAIL_FROM` | Sender email address |
| `SMTP_HOST` | SMTP server (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (typically `587`) |
| `SMTP_USER` | SMTP authentication username |
| `SMTP_PASSWORD` | SMTP password or app password |
| `FUND_NAME` | Display name for the fund (optional) |

The workflow runs Monday–Friday at 13:00 UTC (7:00 AM Guatemala / CST). Trigger manually via `workflow_dispatch` at any time.

## Report Sections

1. **Executive Summary** — 8 KPI cards + cumulative performance chart
2. **Risk Metrics** — Full VaR table, drawdown analysis, rolling VaR chart
3. **Performance Attribution** — Brinson-Fachler sector table
4. **Factor Exposure** — FF5 loadings, R², factor return attribution, heatmap
5. **Stress Testing** — Crisis replay, shock scenarios, rate/recession impacts
6. **Monte Carlo** — Fan chart with 10,000 paths, 126-day horizon
7. **Holdings Detail** — Sector allocation chart + full holdings table
8. **Compliance / GIPS Notes** — Disclosure and data source attribution

## Architecture

```
main.py                  # CLI orchestrator
├── report_generator.py  # Full pipeline: data → metrics → charts → HTML
├── risk_metrics.py      # VaR, CVaR, Sharpe, Sortino, Calmar, Alpha, Beta...
├── factor_analysis.py   # Fama-French 5-factor regression + rolling exposures
├── stress_testing.py    # Historical crisis + shock + macro scenarios
├── attribution.py       # Brinson-Fachler sector attribution
├── chart_generator.py   # Matplotlib charts → base64 PNG
├── template.html        # Jinja2 HTML template (institutional design)
├── config.py            # Tickers, benchmark, risk-free rate, color palette
└── portfolio.csv        # Holdings: ticker, shares, cost_basis, weight
```

## Data Sources

- **Price data**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance`
- **Factor data**: [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (Fama-French 5-Factor Daily)
- **Benchmark**: SPY (SPDR S&P 500 ETF)

## License

MIT
