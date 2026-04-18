import numpy as np
import pandas as pd
from config import SECTOR_MAP


def brinson_fachler(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    sector_map: dict = None,
) -> dict:
    """
    Brinson-Fachler performance attribution at the security level.
    Returns allocation, selection, and interaction effects.
    """
    tickers = portfolio_weights.index.intersection(benchmark_weights.index)
    tickers = tickers.union(portfolio_weights.index).union(benchmark_weights.index)

    wp = portfolio_weights.reindex(tickers, fill_value=0.0)
    wb = benchmark_weights.reindex(tickers, fill_value=0.0)
    rp = portfolio_returns.reindex(tickers, fill_value=0.0)
    rb = benchmark_returns.reindex(tickers, fill_value=0.0)

    rb_total = float((wb * rb).sum())

    allocation_effect = (wp - wb) * (rb - rb_total)
    selection_effect = wb * (rp - rb)
    interaction_effect = (wp - wb) * (rp - rb)
    total_active = allocation_effect + selection_effect + interaction_effect

    results = pd.DataFrame(
        {
            "portfolio_weight": wp,
            "benchmark_weight": wb,
            "portfolio_return": rp,
            "benchmark_return": rb,
            "allocation_effect": allocation_effect,
            "selection_effect": selection_effect,
            "interaction_effect": interaction_effect,
            "total_active": total_active,
        }
    )

    summary = {
        "total_allocation_effect": float(allocation_effect.sum()),
        "total_selection_effect": float(selection_effect.sum()),
        "total_interaction_effect": float(interaction_effect.sum()),
        "total_active_return": float(total_active.sum()),
        "portfolio_return": float((wp * rp).sum()),
        "benchmark_return": rb_total,
        "security_detail": results.to_dict("index"),
    }
    return summary


def sector_attribution(
    portfolio_weights: dict,
    portfolio_returns: dict,
    benchmark_sector_weights: dict = None,
    benchmark_sector_returns: dict = None,
    sector_map: dict = None,
) -> dict:
    """
    Aggregate Brinson-Fachler effects at the sector level.
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    sectors = {}
    for ticker, weight in portfolio_weights.items():
        sector = sector_map.get(ticker, "Other")
        ret = portfolio_returns.get(ticker, 0.0)
        if sector not in sectors:
            sectors[sector] = {"port_weight": 0.0, "port_return_contrib": 0.0, "tickers": []}
        sectors[sector]["port_weight"] += weight
        sectors[sector]["port_return_contrib"] += weight * ret
        sectors[sector]["tickers"].append(ticker)

    for sector in sectors:
        w = sectors[sector]["port_weight"]
        if w > 0:
            sectors[sector]["port_return"] = sectors[sector]["port_return_contrib"] / w
        else:
            sectors[sector]["port_return"] = 0.0

    if benchmark_sector_weights is None:
        spy_sector_weights = {
            "Information Technology": 0.31,
            "Healthcare": 0.13,
            "Financials": 0.13,
            "Consumer Discretionary": 0.10,
            "Industrials": 0.09,
            "Communication Services": 0.09,
            "Consumer Staples": 0.06,
            "Energy": 0.04,
            "Utilities": 0.02,
            "Materials": 0.02,
            "Real Estate": 0.02,
        }
        tech_like = ["Cybersecurity", "Software & Cloud", "E-Commerce"]
        bench_weights = {}
        for sector in sectors:
            if sector in tech_like:
                bench_weights[sector] = spy_sector_weights.get("Information Technology", 0.31) / len(tech_like)
            elif sector == "Aerospace & Defense":
                bench_weights[sector] = spy_sector_weights.get("Industrials", 0.09) * 0.3
            elif sector == "Healthcare & Biotech":
                bench_weights[sector] = spy_sector_weights.get("Healthcare", 0.13)
            else:
                bench_weights[sector] = 0.02
        total = sum(bench_weights.values())
        benchmark_sector_weights = {k: v / total for k, v in bench_weights.items()}

    if benchmark_sector_returns is None:
        benchmark_sector_returns = {s: 0.0 for s in sectors}

    sector_results = []
    bench_total_return = sum(
        benchmark_sector_weights.get(s, 0) * benchmark_sector_returns.get(s, 0)
        for s in sectors
    )

    for sector, data in sectors.items():
        wp = data["port_weight"]
        wb = benchmark_sector_weights.get(sector, 0.0)
        rp = data["port_return"]
        rb = benchmark_sector_returns.get(sector, 0.0)

        alloc = (wp - wb) * (rb - bench_total_return)
        sel = wb * (rp - rb)
        inter = (wp - wb) * (rp - rb)

        sector_results.append({
            "sector": sector,
            "port_weight": wp,
            "bench_weight": wb,
            "port_return": rp,
            "bench_return": rb,
            "allocation_effect": alloc,
            "selection_effect": sel,
            "interaction_effect": inter,
            "total_effect": alloc + sel + inter,
            "tickers": data["tickers"],
        })

    return {
        "sectors": sector_results,
        "total_allocation": sum(r["allocation_effect"] for r in sector_results),
        "total_selection": sum(r["selection_effect"] for r in sector_results),
        "total_interaction": sum(r["interaction_effect"] for r in sector_results),
        "total_active": sum(r["total_effect"] for r in sector_results),
    }
