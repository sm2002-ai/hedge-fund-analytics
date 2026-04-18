import io
import zipfile
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from config import RISK_FREE_RATE


FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


def fetch_fama_french_5_factors(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Download Fama-French 5-factor daily data from Kenneth French Data Library."""
    try:
        resp = requests.get(FF5_URL, timeout=30)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in z.namelist() if n.endswith(".CSV")][0]
        with z.open(csv_name) as f:
            raw = f.read().decode("utf-8", errors="replace")

        lines = raw.split("\n")
        data_lines = []
        in_annual = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Annual"):
                in_annual = True
                continue
            if in_annual:
                continue
            parts = stripped.split(",")
            if len(parts) >= 6:
                try:
                    int(parts[0].strip())
                    data_lines.append(stripped)
                except ValueError:
                    continue

        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
            names=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        )
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df
    except Exception as e:
        print(f"[factor_analysis] Could not fetch FF5 data: {e}. Using synthetic fallback.")
        return _synthetic_ff5(start_date, end_date)


def _synthetic_ff5(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Synthetic FF5 factors for offline/testing use."""
    dates = pd.bdate_range(
        start=start_date or "2023-01-01",
        end=end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, len(dates)),
            "SMB": rng.normal(0.0001, 0.005, len(dates)),
            "HML": rng.normal(0.0001, 0.005, len(dates)),
            "RMW": rng.normal(0.0001, 0.003, len(dates)),
            "CMA": rng.normal(0.0001, 0.003, len(dates)),
            "RF": np.full(len(dates), RISK_FREE_RATE / 252),
        },
        index=dates,
    )
    return df


def run_ff5_regression(portfolio_returns: pd.Series, ff5: pd.DataFrame) -> dict:
    """OLS regression of portfolio excess returns on FF5 factors."""
    merged = pd.concat([portfolio_returns, ff5], axis=1).dropna()
    merged.columns = ["port"] + list(ff5.columns)

    excess_ret = merged["port"] - merged["RF"]
    X = merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)
    model = sm.OLS(excess_ret, X).fit()

    loadings = {
        "alpha": float(model.params["const"]) * 252,
        "mkt_beta": float(model.params["Mkt-RF"]),
        "smb": float(model.params["SMB"]),
        "hml": float(model.params["HML"]),
        "rmw": float(model.params["RMW"]),
        "cma": float(model.params["CMA"]),
        "r_squared": float(model.rsquared),
        "alpha_pvalue": float(model.pvalues["const"]),
        "t_stats": dict(model.tvalues),
        "p_values": dict(model.pvalues),
    }

    factor_means = merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].mean()
    attribution = {
        "mkt_contribution": float(model.params["Mkt-RF"] * factor_means["Mkt-RF"] * 252),
        "smb_contribution": float(model.params["SMB"] * factor_means["SMB"] * 252),
        "hml_contribution": float(model.params["HML"] * factor_means["HML"] * 252),
        "rmw_contribution": float(model.params["RMW"] * factor_means["RMW"] * 252),
        "cma_contribution": float(model.params["CMA"] * factor_means["CMA"] * 252),
        "alpha_contribution": float(model.params["const"]) * 252,
    }

    return {
        "loadings": loadings,
        "attribution": attribution,
        "model_summary": model.summary().as_text(),
        "residuals": model.resid,
    }


def compute_rolling_factor_exposures(
    portfolio_returns: pd.Series, ff5: pd.DataFrame, window: int = 60
) -> pd.DataFrame:
    """Rolling OLS factor exposures over given window."""
    merged = pd.concat([portfolio_returns, ff5], axis=1).dropna()
    merged.columns = ["port"] + list(ff5.columns)
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    records = []
    for end_loc in range(window, len(merged)):
        chunk = merged.iloc[end_loc - window: end_loc]
        excess = chunk["port"] - chunk["RF"]
        X = sm.add_constant(chunk[factors])
        try:
            m = sm.OLS(excess, X).fit()
            row = {"date": chunk.index[-1]}
            row.update({f: float(m.params[f]) for f in factors})
            row["alpha"] = float(m.params["const"]) * 252
            records.append(row)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("date")
    return df
