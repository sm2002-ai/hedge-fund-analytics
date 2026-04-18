import io
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from config import (
    BLOOMBERG_NAVY, BLOOMBERG_GOLD, BLOOMBERG_SLATE, BLOOMBERG_LIGHT, CHART_PALETTE
)

FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQUARE = (8, 6)
FIGSIZE_TALL = (10, 7)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#CCCCCC",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _header_band(ax, title: str):
    ax.set_title(title, loc="left", pad=12, color="white",
                 bbox=dict(facecolor=BLOOMBERG_NAVY, pad=6, boxstyle="round,pad=0.3"))


def cumulative_performance_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_label: str = "Portfolio",
    benchmark_label: str = "SPY",
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    port_cum = (1 + portfolio_returns).cumprod() - 1
    bench_cum = (1 + benchmark_returns).cumprod() - 1

    ax.plot(port_cum.index, port_cum * 100, color=BLOOMBERG_NAVY, lw=2.2, label=portfolio_label, zorder=3)
    ax.plot(bench_cum.index, bench_cum * 100, color=BLOOMBERG_GOLD, lw=1.8, linestyle="--",
            label=benchmark_label, zorder=2)
    ax.fill_between(port_cum.index, port_cum * 100, bench_cum * 100,
                    where=port_cum >= bench_cum,
                    alpha=0.15, color=BLOOMBERG_NAVY, label="Outperformance")
    ax.fill_between(port_cum.index, port_cum * 100, bench_cum * 100,
                    where=port_cum < bench_cum,
                    alpha=0.15, color="#C73E1D", label="Underperformance")

    ax.set_title("Cumulative Performance vs Benchmark", loc="left", fontsize=13, fontweight="bold",
                 color=BLOOMBERG_NAVY)
    ax.set_ylabel("Return (%)")
    ax.axhline(0, color="#999999", lw=0.8, linestyle=":")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    fig.tight_layout()
    return _to_base64(fig)


def drawdown_chart(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    cum_port = (1 + portfolio_returns).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    dd_port = (cum_port - cum_port.cummax()) / cum_port.cummax() * 100
    dd_bench = (cum_bench - cum_bench.cummax()) / cum_bench.cummax() * 100

    ax.fill_between(dd_port.index, dd_port, 0, alpha=0.7, color=BLOOMBERG_NAVY, label="Portfolio")
    ax.fill_between(dd_bench.index, dd_bench, 0, alpha=0.4, color=BLOOMBERG_GOLD, label="SPY")

    ax.set_title("Drawdown Analysis", loc="left", fontsize=13, fontweight="bold", color=BLOOMBERG_NAVY)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    return _to_base64(fig)


def var_distribution_chart(portfolio_returns: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    returns_pct = portfolio_returns.dropna() * 100
    ax.hist(returns_pct, bins=60, color=BLOOMBERG_NAVY, alpha=0.75, edgecolor="white", lw=0.4)

    var95 = np.percentile(returns_pct, 5)
    var99 = np.percentile(returns_pct, 1)
    ax.axvline(var95, color=BLOOMBERG_GOLD, lw=2, linestyle="--", label=f"VaR 95%: {var95:.2f}%")
    ax.axvline(var99, color="#C73E1D", lw=2, linestyle="--", label=f"VaR 99%: {var99:.2f}%")
    ax.axvline(0, color="#999999", lw=0.8, linestyle=":")

    ax.set_title("Daily Return Distribution & VaR", loc="left", fontsize=13, fontweight="bold",
                 color=BLOOMBERG_NAVY)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    return _to_base64(fig)


def factor_exposure_heatmap(factor_loadings: dict) -> str:
    factors = ["mkt_beta", "smb", "hml", "rmw", "cma"]
    labels = ["Market β", "SMB\n(Size)", "HML\n(Value)", "RMW\n(Profit.)", "CMA\n(Invest.)"]
    values = np.array([[factor_loadings.get(f, 0.0) for f in factors]])

    fig, ax = plt.subplots(figsize=(10, 2.5))
    cmap = plt.cm.RdBu_r
    vmax = max(abs(v) for v in values.flatten()) + 0.1
    im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Factor Loading")

    for j, val in enumerate(values[0]):
        color = "white" if abs(val) > vmax * 0.5 else "black"
        ax.text(j, 0, f"{val:.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title("Fama-French 5-Factor Exposures", loc="left", fontsize=13, fontweight="bold",
                 color=BLOOMBERG_NAVY)
    fig.tight_layout()
    return _to_base64(fig)


def sector_allocation_chart(sector_weights: dict) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sectors = list(sector_weights.keys())
    weights = [sector_weights[s] * 100 for s in sectors]
    colors = CHART_PALETTE[:len(sectors)]

    wedges, texts, autotexts = ax1.pie(
        weights, labels=sectors, autopct="%1.1f%%", colors=colors,
        startangle=140, pctdistance=0.75, textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
        at.set_color("white")
    ax1.set_title("Sector Allocation", fontsize=13, fontweight="bold", color=BLOOMBERG_NAVY, loc="left")

    bars = ax2.barh(sectors, weights, color=colors, edgecolor="white", lw=0.5)
    ax2.set_xlabel("Weight (%)")
    ax2.set_title("Sector Weights", fontsize=13, fontweight="bold", color=BLOOMBERG_NAVY, loc="left")
    ax2.set_xlim(0, max(weights) * 1.15)
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout(pad=2.0)
    return _to_base64(fig)


def stress_test_waterfall_chart(stress_results: dict) -> str:
    labels = []
    impacts = []

    crises = stress_results.get("historical_crises", {})
    for name, data in crises.items():
        if data.get("portfolio_return") is not None:
            labels.append(name[:20])
            impacts.append(data["portfolio_return"] * 100)

    shocks = stress_results.get("single_day_shocks", {})
    for name, data in shocks.items():
        labels.append(name)
        impacts.append(data["portfolio_impact_pct"])

    rh = stress_results.get("rate_hike_100bps", {})
    if rh:
        labels.append("+100bps Rate Hike")
        impacts.append(rh.get("estimated_portfolio_impact_pct", 0))

    rec = stress_results.get("recession_mild", {})
    if rec:
        labels.append("Mild Recession")
        impacts.append(rec.get("estimated_price_impact_pct", 0))

    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    colors = [BLOOMBERG_NAVY if v >= 0 else "#C73E1D" for v in impacts]
    bars = ax.barh(labels, impacts, color=colors, edgecolor="white", lw=0.5)
    ax.axvline(0, color="#555555", lw=1.2)
    ax.set_xlabel("Estimated Portfolio Impact (%)")
    ax.set_title("Stress Test Scenarios", loc="left", fontsize=13, fontweight="bold", color=BLOOMBERG_NAVY)

    for bar, v in zip(bars, impacts):
        xpos = bar.get_width() + (0.3 if v >= 0 else -0.3)
        ha = "left" if v >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha=ha, fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    return _to_base64(fig)


def monte_carlo_fan_chart(portfolio_returns: pd.Series, n_sims: int = 5000, horizon: int = 126) -> str:
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    rng = np.random.default_rng(42)
    sims = rng.normal(mu, sigma, (n_sims, horizon))
    paths = np.cumprod(1 + sims, axis=1)

    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    x = np.arange(horizon)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.fill_between(x, (p5 - 1) * 100, (p95 - 1) * 100, alpha=0.15, color=BLOOMBERG_NAVY, label="5th–95th pct")
    ax.fill_between(x, (p25 - 1) * 100, (p75 - 1) * 100, alpha=0.30, color=BLOOMBERG_NAVY, label="25th–75th pct")
    ax.plot(x, (p50 - 1) * 100, color=BLOOMBERG_NAVY, lw=2, label="Median")
    ax.axhline(0, color="#999999", lw=0.8, linestyle=":")

    ax.set_title(f"Monte Carlo Fan Chart — {horizon}-Day Horizon ({n_sims:,} Simulations)",
                 loc="left", fontsize=13, fontweight="bold", color=BLOOMBERG_NAVY)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Projected Return (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    return _to_base64(fig)


def rolling_var_chart(rolling_var_30: pd.Series, rolling_var_60: pd.Series, rolling_var_90: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(rolling_var_30.index, rolling_var_30 * 100, color=CHART_PALETTE[0], lw=1.5, label="30-day")
    ax.plot(rolling_var_60.index, rolling_var_60 * 100, color=CHART_PALETTE[1], lw=1.5, label="60-day")
    ax.plot(rolling_var_90.index, rolling_var_90 * 100, color=CHART_PALETTE[2], lw=1.5, label="90-day")
    ax.set_title("Rolling VaR (95% Historical Simulation)", loc="left", fontsize=13, fontweight="bold",
                 color=BLOOMBERG_NAVY)
    ax.set_ylabel("Daily VaR (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    return _to_base64(fig)


def generate_all_charts(portfolio_returns, benchmark_returns, risk_metrics, factor_data,
                         stress_data, sector_weights) -> dict:
    charts = {}
    charts["cumulative_performance"] = cumulative_performance_chart(portfolio_returns, benchmark_returns)
    charts["drawdown"] = drawdown_chart(portfolio_returns, benchmark_returns)
    charts["var_distribution"] = var_distribution_chart(portfolio_returns)
    charts["monte_carlo"] = monte_carlo_fan_chart(portfolio_returns)

    if factor_data and "loadings" in factor_data:
        charts["factor_heatmap"] = factor_exposure_heatmap(factor_data["loadings"])

    if sector_weights:
        charts["sector_allocation"] = sector_allocation_chart(sector_weights)

    if stress_data:
        charts["stress_test"] = stress_test_waterfall_chart(stress_data)

    if risk_metrics.get("rolling_var_30") is not None:
        charts["rolling_var"] = rolling_var_chart(
            risk_metrics["rolling_var_30"].dropna(),
            risk_metrics["rolling_var_60"].dropna(),
            risk_metrics["rolling_var_90"].dropna(),
        )

    return charts
