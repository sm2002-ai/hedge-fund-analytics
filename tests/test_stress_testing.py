import numpy as np
import pandas as pd
import pytest

import stress_testing
from stress_testing import (
    SHOCK_SCENARIOS,
    run_recession_scenario,
    run_single_day_shocks,
)


class TestSingleDayShocks:
    def test_all_scenarios_present(self):
        results = run_single_day_shocks(portfolio_beta=1.0)
        assert set(results.keys()) == set(SHOCK_SCENARIOS.keys())

    def test_impact_scales_linearly_with_beta(self):
        at_one = run_single_day_shocks(1.0)
        at_two = run_single_day_shocks(2.0)
        for name in SHOCK_SCENARIOS:
            assert at_two[name]["portfolio_impact"] == pytest.approx(
                2.0 * at_one[name]["portfolio_impact"]
            )

    def test_impact_pct_matches_impact_times_100(self):
        results = run_single_day_shocks(portfolio_beta=1.2)
        for name, data in results.items():
            assert data["portfolio_impact_pct"] == pytest.approx(data["portfolio_impact"] * 100)

    def test_larger_shock_yields_larger_loss(self):
        results = run_single_day_shocks(1.0)
        impacts = [results[name]["portfolio_impact"] for name in ["Market -5%", "Market -10%", "Market -20%"]]
        assert impacts[0] > impacts[1] > impacts[2]  # more negative


class TestRecessionScenario:
    def test_output_has_expected_keys(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-03", periods=252, freq="B")
        returns = pd.Series(rng.normal(0.001, 0.01, 252), index=idx)
        res = run_recession_scenario(returns, earnings_decline_pct=0.20)
        for k in [
            "assumed_earnings_decline",
            "estimated_price_impact",
            "estimated_price_impact_pct",
            "zscore_equivalent",
        ]:
            assert k in res

    def test_larger_decline_yields_more_negative_impact(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-03", periods=252, freq="B")
        returns = pd.Series(rng.normal(0.001, 0.01, 252), index=idx)
        mild = run_recession_scenario(returns, 0.10)
        severe = run_recession_scenario(returns, 0.25)
        assert severe["estimated_price_impact"] < mild["estimated_price_impact"]

    def test_price_impact_uses_pe_multiplier(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-03", periods=252, freq="B")
        returns = pd.Series(rng.normal(0.001, 0.01, 252), index=idx)
        res = run_recession_scenario(returns, 0.20)
        # avg_pe_sensitivity = 0.8, so price decline = -(0.20 * 0.8) = -0.16
        assert res["estimated_price_impact"] == pytest.approx(-0.16)

    def test_zero_volatility_does_not_blow_up(self):
        # epsilon guard (1e-9) should keep zscore finite
        idx = pd.date_range("2022-01-03", periods=50, freq="B")
        returns = pd.Series(0.0, index=idx)
        res = run_recession_scenario(returns, 0.20)
        assert np.isfinite(res["zscore_equivalent"])


class TestHistoricalCrisis:
    """Uses yfinance — mock the download to avoid network."""

    def test_returns_all_crisis_periods(self, mocker):
        # Build a fake SPY time series covering all crisis windows.
        idx = pd.date_range("1999-01-01", "2023-12-31", freq="B")
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))), index=idx)
        fake_spy = pd.DataFrame({"Close": prices})
        mocker.patch.object(stress_testing.yf, "download", return_value=fake_spy)

        port_idx = pd.date_range("2007-01-01", "2023-12-31", freq="B")
        port_returns = pd.Series(rng.normal(0.0005, 0.012, len(port_idx)), index=port_idx)

        results = stress_testing.run_historical_crisis(port_returns, {"AAA": 1.0})
        assert set(results.keys()) == set(stress_testing.CRISIS_PERIODS.keys())

    def test_empty_crisis_window_returns_nones(self, mocker):
        # Fake SPY that doesn't overlap any crisis
        idx = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        fake_spy = pd.DataFrame({"Close": pd.Series(100.0, index=idx)})
        mocker.patch.object(stress_testing.yf, "download", return_value=fake_spy)

        port_returns = pd.Series(0.0, index=idx)
        results = stress_testing.run_historical_crisis(port_returns, {"AAA": 1.0})
        # All windows are pre-2024 and thus empty
        for name, data in results.items():
            assert data["spy_return"] is None


class TestRateHikeScenario:
    def test_uses_linear_scaling_in_bps(self, mocker):
        # Patch TLT download so we can exercise the cov-based path deterministically.
        idx = pd.date_range("2022-01-03", periods=300, freq="B")
        rng = np.random.default_rng(0)
        tlt_prices = 100 * np.cumprod(1 + rng.normal(-0.0002, 0.008, len(idx)))
        fake_tlt = pd.DataFrame({"Close": pd.Series(tlt_prices, index=idx)})
        mocker.patch.object(stress_testing.yf, "download", return_value=fake_tlt)

        port = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
        bench = pd.Series(rng.normal(0.0003, 0.009, len(idx)), index=idx)

        r100 = stress_testing.run_rate_hike_scenario(port, bench, 100)
        r200 = stress_testing.run_rate_hike_scenario(port, bench, 200)
        # Portfolio impact scales linearly with bps
        assert r200["estimated_portfolio_impact"] == pytest.approx(
            2.0 * r100["estimated_portfolio_impact"]
        )
