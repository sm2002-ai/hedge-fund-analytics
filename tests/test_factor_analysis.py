import numpy as np
import pandas as pd
import pytest
import requests

import factor_analysis
from factor_analysis import (
    _synthetic_ff5,
    compute_rolling_factor_exposures,
    fetch_fama_french_5_factors,
    run_ff5_regression,
)


class TestSyntheticFF5:
    def test_has_expected_columns(self):
        df = _synthetic_ff5("2023-01-01", "2023-03-01")
        assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

    def test_deterministic_seed(self):
        a = _synthetic_ff5("2023-01-01", "2023-03-01")
        b = _synthetic_ff5("2023-01-01", "2023-03-01")
        pd.testing.assert_frame_equal(a, b)

    def test_date_range_respected(self):
        df = _synthetic_ff5("2023-01-02", "2023-01-31")
        assert df.index.min() >= pd.Timestamp("2023-01-02")
        assert df.index.max() <= pd.Timestamp("2023-01-31")

    def test_rf_is_constant_daily_rate(self):
        df = _synthetic_ff5("2023-01-01", "2023-03-01")
        assert df["RF"].nunique() == 1


class TestFetchFamaFrench:
    def test_falls_back_to_synthetic_on_network_error(self, mocker):
        mocker.patch.object(
            factor_analysis.requests, "get",
            side_effect=requests.ConnectionError("no network"),
        )
        df = fetch_fama_french_5_factors("2023-01-01", "2023-02-01")
        # Should have synthetic data with the right columns and range
        assert set(df.columns) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}
        assert len(df) > 0

    def test_falls_back_to_synthetic_on_http_error(self, mocker):
        class FakeResp:
            status_code = 500
            content = b""
            def raise_for_status(self):
                raise requests.HTTPError("server error")
        mocker.patch.object(factor_analysis.requests, "get", return_value=FakeResp())
        df = fetch_fama_french_5_factors("2023-01-01", "2023-02-01")
        assert not df.empty
        assert "Mkt-RF" in df.columns


class TestRunFF5Regression:
    def test_output_structure(self):
        ff5 = _synthetic_ff5("2022-01-01", "2023-01-01")
        rng = np.random.default_rng(0)
        port = pd.Series(rng.normal(0.001, 0.01, len(ff5)), index=ff5.index)
        result = run_ff5_regression(port, ff5)
        assert "loadings" in result
        assert "attribution" in result
        for key in ["alpha", "mkt_beta", "smb", "hml", "rmw", "cma", "r_squared"]:
            assert key in result["loadings"]

    def test_recovers_known_beta(self):
        ff5 = _synthetic_ff5("2022-01-01", "2023-06-01")
        # Construct port = RF + 1.5*Mkt-RF + noise => mkt_beta ≈ 1.5
        rng = np.random.default_rng(0)
        port = ff5["RF"] + 1.5 * ff5["Mkt-RF"] + rng.normal(0, 0.0005, len(ff5))
        result = run_ff5_regression(port, ff5)
        assert result["loadings"]["mkt_beta"] == pytest.approx(1.5, abs=0.1)

    def test_r_squared_in_unit_interval(self):
        ff5 = _synthetic_ff5("2022-01-01", "2023-01-01")
        rng = np.random.default_rng(1)
        port = pd.Series(rng.normal(0.001, 0.01, len(ff5)), index=ff5.index)
        r2 = run_ff5_regression(port, ff5)["loadings"]["r_squared"]
        assert 0.0 <= r2 <= 1.0


class TestRollingFactorExposures:
    def test_returns_empty_when_insufficient_data(self):
        ff5 = _synthetic_ff5("2023-01-01", "2023-02-01")  # <60 rows
        port = pd.Series(0.001, index=ff5.index)
        out = compute_rolling_factor_exposures(port, ff5, window=60)
        assert out.empty

    def test_has_one_row_per_window_end(self):
        ff5 = _synthetic_ff5("2022-01-01", "2023-01-01")
        port = pd.Series(0.001, index=ff5.index)
        window = 60
        aligned = pd.concat([port, ff5], axis=1).dropna()
        out = compute_rolling_factor_exposures(port, ff5, window=window)
        assert len(out) == max(0, len(aligned) - window)

    def test_output_has_expected_columns(self):
        ff5 = _synthetic_ff5("2022-01-01", "2023-01-01")
        port = pd.Series(0.001, index=ff5.index)
        out = compute_rolling_factor_exposures(port, ff5, window=60)
        for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "alpha"]:
            assert col in out.columns
