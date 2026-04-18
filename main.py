#!/usr/bin/env python3
"""
Hedge Fund Analytics — Main Orchestrator

Usage:
    python main.py
    python main.py --portfolio portfolio.csv --benchmark SPY --email user@email.com
    python main.py --lookback 126 --output custom_report.html --no-email
"""
import argparse
import os
import sys
import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate institutional hedge fund analytics report.")
    parser.add_argument("--portfolio", default="portfolio.csv", help="Path to portfolio CSV file")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--lookback", type=int, default=252, help="Trading days of history (default: 252)")
    parser.add_argument("--output", default=None, help="Output HTML path (default: report_YYYY-MM-DD.html)")
    parser.add_argument("--email", default=None, help="Override recipient email address")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.portfolio):
        print(f"ERROR: Portfolio file not found: {args.portfolio}")
        sys.exit(1)

    portfolio_df = pd.read_csv(args.portfolio)
    required_cols = {"ticker", "shares", "cost_basis", "weight"}
    missing = required_cols - set(portfolio_df.columns)
    if missing:
        print(f"ERROR: Portfolio CSV missing columns: {missing}")
        sys.exit(1)

    if args.benchmark != "SPY":
        import config as cfg
        cfg.BENCHMARK = args.benchmark

    if args.email:
        import config as cfg
        cfg.EMAIL_TO = args.email

    output_path = args.output or f"report_{datetime.date.today().strftime('%Y-%m-%d')}.html"

    print("=" * 60)
    print(f"  Hedge Fund Analytics Report Generator")
    print(f"  Portfolio: {args.portfolio} ({len(portfolio_df)} holdings)")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Lookback:  {args.lookback} trading days")
    print(f"  Output:    {output_path}")
    print("=" * 60)

    from report_generator import generate_report, send_email_report

    html = generate_report(portfolio_df, lookback_days=args.lookback, output_path=output_path)

    if not args.no_email:
        try:
            send_email_report(html, attachment_path=output_path)
        except Exception as e:
            print(f"Email delivery failed: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
