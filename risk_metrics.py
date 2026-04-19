import numpy as np
import pandas as pd
from scipy import stats
from config import RISK_FREE_RATE, CONFIDENCE_LEVELS


def compute_var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR at given confidence level. Returns positive loss number."""
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def compute_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation VaR. Returns positive loss number."""
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall. Returns positive loss number."""
    var = compute_var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if tail.empty:
        return var
    return -tail.mean()


def compute_rolling_var(returns: pd.Series, window: int = 30, confidence: float = 0.95) -> pd.Series:
    """Rolling VaR using historical simulation over given window."""
    return returns.rolling(window).apply(
        lambda x: compute_var_historical(pd.Series(x), confidence), raw=False
    )


def compute_sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - rf / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_sortino(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino ratio using downside deviation."""
    excess = returns - rf / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def compute_max_drawdown(returns: pd.Series) -> dict:
    """Maximum drawdown with peak, trough, and recovery info."""
    returns = returns.dropna()
    if returns.empty:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None,
                "recovery_date": None, "recovery_days": None,
                "drawdown_series": pd.Series(dtype=float)}
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max

    max_dd = float(drawdown.min())
    trough_idx = drawdown.idxmin()
    peak_idx = rolling_max[:trough_idx].idxmax()

    recovered = cum[trough_idx:][cum[trough_idx:] >= rolling_max[trough_idx]]
    recovery_idx = recovered.index[0] if not recovered.empty else None
    recovery_days = (recovery_idx - trough_idx).days if recovery_idx else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_idx.strftime("%Y-%m-%d") if hasattr(peak_idx, "strftime") else str(peak_idx),
        "trough_date": trough_idx.strftime("%Y-%m-%d") if hasattr(trough_idx, "strftime") else str(trough_idx),
        "recovery_date": recovery_idx.strftime("%Y-%m-%d") if recovery_idx and hasattr(recovery_idx, "strftime") else None,
        "recovery_days": recovery_days,
        "drawdown_series": drawdown,
    }


def compute_calmar(returns: pd.Series) -> float:
    """Calmar ratio: annualized return / abs(max drawdown)."""
    mdd = compute_max_drawdown(returns)["max_drawdown"]
    ann_ret = (1 + returns.mean()) ** 252 - 1
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def compute_beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """OLS beta and annualized alpha vs benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["port", "bench"]
    cov = np.cov(aligned["port"], aligned["bench"])
    beta = cov[0, 1] / cov[1, 1]
    alpha_daily = aligned["port"].mean() - beta * aligned["bench"].mean()
    alpha_annual = alpha_daily * 252
    return {"beta": float(beta), "alpha_annual": float(alpha_annual)}


def compute_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Information ratio: annualized active return / tracking error."""
    active = portfolio_returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(252))


def compute_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized tracking error."""
    active = (portfolio_returns - benchmark_returns).dropna()
    return float(active.std() * np.sqrt(252))


def compute_win_loss(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Win/loss ratio vs benchmark (days portfolio outperforms)."""
    active = (portfolio_returns - benchmark_returns).dropna()
    wins = (active > 0).sum()
    losses = (active <= 0).sum()
    win_rate = wins / len(active) if len(active) > 0 else 0
    avg_win = active[active > 0].mean() if wins > 0 else 0
    avg_loss = active[active <= 0].mean() if losses > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    return {
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
    }


def compute_all_risk_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Compute the full risk metric suite."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    port = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    mdd_data = compute_max_drawdown(port)
    ba = compute_beta_alpha(port, bench)

    metrics = {
        "var_95_parametric": compute_var_parametric(port, 0.95),
        "var_99_parametric": compute_var_parametric(port, 0.99),
        "var_95_historical": compute_var_historical(port, 0.95),
        "var_99_historical": compute_var_historical(port, 0.99),
        "cvar_95": compute_cvar(port, 0.95),
        "cvar_99": compute_cvar(port, 0.99),
        "rolling_var_30": compute_rolling_var(port, 30),
        "rolling_var_60": compute_rolling_var(port, 60),
        "rolling_var_90": compute_rolling_var(port, 90),
        "sharpe": compute_sharpe(port),
        "sortino": compute_sortino(port),
        "calmar": compute_calmar(port),
        "information_ratio": compute_information_ratio(port, bench),
        "max_drawdown": mdd_data["max_drawdown"],
        "max_drawdown_peak": mdd_data["peak_date"],
        "max_drawdown_trough": mdd_data["trough_date"],
        "max_drawdown_recovery": mdd_data["recovery_date"],
        "max_drawdown_recovery_days": mdd_data["recovery_days"],
        "drawdown_series": mdd_data["drawdown_series"],
        "beta": ba["beta"],
        "alpha_annual": ba["alpha_annual"],
        "tracking_error": compute_tracking_error(port, bench),
        "win_loss": compute_win_loss(port, bench),
        "annualized_return": float((1 + port.mean()) ** 252 - 1),
        "annualized_volatility": float(port.std() * np.sqrt(252)),
        "total_return": float((1 + port).prod() - 1),
        "benchmark_return": float((1 + bench).prod() - 1),
    }
    return metrics
