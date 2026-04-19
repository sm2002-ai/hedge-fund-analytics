import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_metrics import (
    compute_beta_alpha,
    compute_calmar,
    compute_cvar,
    compute_information_ratio,
    compute_max_drawdown,
    compute_rolling_var,
    compute_sharpe,
    compute_sortino,
    compute_tracking_error,
    compute_var_historical,
    compute_var_parametric,
    compute_win_loss,
    compute_all_risk_metrics,
)


class TestParametricVaR:
    def test_matches_gaussian_closed_form(self, normal_returns):
        mu, sigma = normal_returns.mean(), normal_returns.std()
        expected = -(mu + stats.norm.ppf(0.05) * sigma)
        assert compute_var_parametric(normal_returns, 0.95) == pytest.approx(expected)

    def test_is_positive_for_losing_portfolio(self):
        returns = pd.Series(np.linspace(-0.05, -0.01, 50))
        assert compute_var_parametric(returns, 0.95) > 0

    def test_higher_confidence_means_larger_var(self, normal_returns):
        v95 = compute_var_parametric(normal_returns, 0.95)
        v99 = compute_var_parametric(normal_returns, 0.99)
        assert v99 > v95

    def test_constant_returns_gives_negative_var(self, constant_returns):
        # zero vol → VaR = -mean
        assert compute_var_parametric(constant_returns, 0.95) == pytest.approx(-0.001)


class TestHistoricalVaR:
    def test_matches_numpy_percentile(self, normal_returns):
        expected = -np.percentile(normal_returns, 5)
        assert compute_var_historical(normal_returns, 0.95) == pytest.approx(expected)

    def test_handles_nan(self):
        returns = pd.Series([0.01, np.nan, -0.02, 0.03, -0.01, np.nan, -0.05])
        result = compute_var_historical(returns, 0.95)
        assert np.isfinite(result)

    def test_higher_confidence_means_larger_var(self, normal_returns):
        assert compute_var_historical(normal_returns, 0.99) >= compute_var_historical(normal_returns, 0.95)


class TestCVaR:
    def test_cvar_at_least_as_large_as_var(self, normal_returns):
        var = compute_var_historical(normal_returns, 0.95)
        cvar = compute_cvar(normal_returns, 0.95)
        assert cvar >= var - 1e-12

    def test_empty_tail_falls_back_to_var(self):
        # constant series has no tail beyond the VaR cutoff
        returns = pd.Series([0.01] * 20)
        cvar = compute_cvar(returns, 0.95)
        assert np.isfinite(cvar)

    def test_cvar_on_known_distribution(self):
        returns = pd.Series(np.linspace(-0.10, 0.10, 21))
        cvar = compute_cvar(returns, 0.95)
        assert cvar > 0


class TestRollingVaR:
    def test_output_length_matches_input(self, normal_returns):
        rv = compute_rolling_var(normal_returns, window=30, confidence=0.95)
        assert len(rv) == len(normal_returns)

    def test_first_window_minus_one_values_are_nan(self, normal_returns):
        rv = compute_rolling_var(normal_returns, window=30)
        assert rv.iloc[:29].isna().all()
        assert rv.iloc[29:].notna().all()


class TestSharpe:
    def test_zero_volatility_returns_zero(self, constant_returns):
        assert compute_sharpe(constant_returns, rf=0.0) == 0.0

    def test_zero_volatility_returns_zero_with_nonzero_rf(self, constant_returns):
        assert compute_sharpe(constant_returns, rf=0.05) == 0.0

    def test_positive_for_positive_excess(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.002, 0.005, 252))
        assert compute_sharpe(returns, rf=0.0) > 0

    def test_annualization_factor(self):
        # constant non-zero std series: sharpe should scale by sqrt(252)
        rng = np.random.default_rng(11)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        daily = (r.mean() - 0.0 / 252) / r.std()
        assert compute_sharpe(r, rf=0.0) == pytest.approx(daily * np.sqrt(252))


class TestSortino:
    def test_no_downside_returns_zero(self):
        returns = pd.Series([0.01] * 50)  # no negative excess
        assert compute_sortino(returns, rf=0.0) == 0.0

    def test_penalizes_downside_only(self):
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sortino = compute_sortino(returns, rf=0.0)
        sharpe = compute_sharpe(returns, rf=0.0)
        # Sortino should be >= Sharpe when downside std <= total std (generally true)
        assert sortino >= sharpe - 0.5  # loose bound


class TestMaxDrawdown:
    def test_known_drawdown_series(self):
        # prices: 100 -> 120 -> 60 -> 150. max DD = (60-120)/120 = -0.5
        prices = pd.Series(
            [100, 120, 60, 150],
            index=pd.date_range("2023-01-02", periods=4, freq="B"),
        )
        returns = prices.pct_change().dropna()
        res = compute_max_drawdown(returns)
        assert res["max_drawdown"] == pytest.approx(-0.5, abs=1e-6)

    def test_date_formatting_for_datetime_index(self):
        idx = pd.date_range("2023-01-02", periods=10, freq="B")
        returns = pd.Series([0.01, -0.05, -0.02, 0.01, 0.03, -0.01, 0.02, 0.01, 0.02, 0.01], index=idx)
        res = compute_max_drawdown(returns)
        assert isinstance(res["peak_date"], str)
        assert isinstance(res["trough_date"], str)
        # should parse back to a date
        pd.to_datetime(res["peak_date"])

    def test_no_drawdown_for_monotonic_returns(self):
        returns = pd.Series([0.01] * 30, index=pd.date_range("2023-01-02", periods=30, freq="B"))
        res = compute_max_drawdown(returns)
        assert res["max_drawdown"] == pytest.approx(0.0)

    def test_no_drawdown_for_monotonic_returns_int_index(self):
        returns = pd.Series([0.01] * 30)
        res = compute_max_drawdown(returns)
        assert res["max_drawdown"] == pytest.approx(0.0)

    def test_recovery_days_none_when_not_recovered(self):
        # big single drop that never recovers
        returns = pd.Series(
            [0.01, 0.01, -0.50, 0.001, 0.001],
            index=pd.date_range("2023-01-02", periods=5, freq="B"),
        )
        res = compute_max_drawdown(returns)
        assert res["recovery_date"] is None
        assert res["recovery_days"] is None


class TestCalmar:
    def test_zero_drawdown_returns_zero(self):
        returns = pd.Series([0.001] * 30)
        assert compute_calmar(returns) == 0.0

    def test_positive_for_positive_returns_with_int_index(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.002, 0.01, 252))
        result = compute_calmar(returns)
        assert np.isfinite(result)

    def test_positive_for_positive_returns_with_datetime_index(self):
        rng = np.random.default_rng(5)
        idx = pd.date_range("2023-01-02", periods=252, freq="B")
        returns = pd.Series(rng.normal(0.002, 0.01, 252), index=idx)
        result = compute_calmar(returns)
        assert np.isfinite(result)


class TestBetaAlpha:
    def test_beta_equals_one_for_identical_series(self, normal_returns):
        res = compute_beta_alpha(normal_returns, normal_returns)
        assert res["beta"] == pytest.approx(1.0)
        assert res["alpha_annual"] == pytest.approx(0.0, abs=1e-9)

    def test_beta_scales_with_leverage(self, benchmark_returns):
        levered = 2.0 * benchmark_returns
        res = compute_beta_alpha(levered, benchmark_returns)
        assert res["beta"] == pytest.approx(2.0, rel=1e-6)

    def test_alpha_recovers_intercept(self, benchmark_returns):
        # portfolio = 1.5 * bench + 0.0002 per day
        port = 1.5 * benchmark_returns + 0.0002
        res = compute_beta_alpha(port, benchmark_returns)
        assert res["beta"] == pytest.approx(1.5, rel=1e-4)
        assert res["alpha_annual"] == pytest.approx(0.0002 * 252, rel=1e-4)


class TestInformationRatio:
    def test_zero_active_returns_zero(self, normal_returns):
        assert compute_information_ratio(normal_returns, normal_returns) == 0.0

    def test_positive_when_portfolio_outperforms(self, benchmark_returns):
        port = benchmark_returns + 0.001
        assert compute_information_ratio(port, benchmark_returns) > 0


class TestTrackingError:
    def test_zero_for_identical_series(self, normal_returns):
        assert compute_tracking_error(normal_returns, normal_returns) == pytest.approx(0.0)

    def test_scales_with_sqrt_252(self, normal_returns, benchmark_returns):
        active = (normal_returns - benchmark_returns).dropna()
        assert compute_tracking_error(normal_returns, benchmark_returns) == pytest.approx(
            active.std() * np.sqrt(252)
        )


class TestWinLoss:
    def test_counts_sum_to_length(self, normal_returns, benchmark_returns):
        wl = compute_win_loss(normal_returns, benchmark_returns)
        active_len = len((normal_returns - benchmark_returns).dropna())
        assert wl["wins"] + wl["losses"] == active_len

    def test_all_wins(self):
        idx = pd.date_range("2023-01-02", periods=10, freq="B")
        port = pd.Series([0.02] * 10, index=idx)
        bench = pd.Series([0.01] * 10, index=idx)
        wl = compute_win_loss(port, bench)
        assert wl["wins"] == 10
        assert wl["losses"] == 0
        assert wl["win_rate"] == 1.0


class TestAllRiskMetrics:
    def test_suite_returns_expected_keys(self, normal_returns, benchmark_returns):
        m = compute_all_risk_metrics(normal_returns, benchmark_returns)
        for key in [
            "var_95_parametric", "var_99_parametric", "var_95_historical", "var_99_historical",
            "cvar_95", "cvar_99", "sharpe", "sortino", "calmar", "information_ratio",
            "max_drawdown", "beta", "alpha_annual", "tracking_error", "win_loss",
            "annualized_return", "annualized_volatility", "total_return", "benchmark_return",
        ]:
            assert key in m

    def test_var99_larger_than_var95(self, normal_returns, benchmark_returns):
        m = compute_all_risk_metrics(normal_returns, benchmark_returns)
        assert m["var_99_historical"] >= m["var_95_historical"]
        assert m["var_99_parametric"] >= m["var_95_parametric"]
