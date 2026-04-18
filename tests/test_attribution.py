import pandas as pd
import pytest

from attribution import brinson_fachler, sector_attribution


class TestBrinsonFachler:
    def test_components_sum_to_total_active(self):
        # Toy: 2 securities, portfolio overweights A and its A return beats bench.
        pw = pd.Series({"A": 0.7, "B": 0.3})
        bw = pd.Series({"A": 0.5, "B": 0.5})
        pr = pd.Series({"A": 0.10, "B": 0.02})
        br = pd.Series({"A": 0.08, "B": 0.04})

        res = brinson_fachler(pw, bw, pr, br)
        summed = (
            res["total_allocation_effect"]
            + res["total_selection_effect"]
            + res["total_interaction_effect"]
        )
        assert summed == pytest.approx(res["total_active_return"])

    def test_total_active_matches_port_minus_bench(self):
        pw = pd.Series({"A": 0.7, "B": 0.3})
        bw = pd.Series({"A": 0.5, "B": 0.5})
        pr = pd.Series({"A": 0.10, "B": 0.02})
        br = pd.Series({"A": 0.08, "B": 0.04})

        res = brinson_fachler(pw, bw, pr, br)
        expected_active = res["portfolio_return"] - res["benchmark_return"]
        assert res["total_active_return"] == pytest.approx(expected_active)

    def test_identical_portfolio_and_benchmark_has_zero_active(self):
        pw = pd.Series({"A": 0.4, "B": 0.6})
        pr = pd.Series({"A": 0.05, "B": 0.02})
        res = brinson_fachler(pw, pw, pr, pr)
        assert res["total_active_return"] == pytest.approx(0.0)
        assert res["total_allocation_effect"] == pytest.approx(0.0)
        assert res["total_selection_effect"] == pytest.approx(0.0)
        assert res["total_interaction_effect"] == pytest.approx(0.0)

    def test_ticker_in_bench_not_portfolio_gets_zero_weight(self):
        pw = pd.Series({"A": 1.0})
        bw = pd.Series({"A": 0.5, "B": 0.5})
        pr = pd.Series({"A": 0.05})
        br = pd.Series({"A": 0.03, "B": 0.01})

        res = brinson_fachler(pw, bw, pr, br)
        detail = res["security_detail"]
        assert detail["B"]["portfolio_weight"] == 0.0
        # Portfolio return: 1.0 * 0.05 = 0.05
        assert res["portfolio_return"] == pytest.approx(0.05)

    def test_selection_effect_uses_benchmark_weight(self):
        # If portfolio and benchmark weights are equal, allocation and
        # interaction should be zero and all active should come from selection.
        pw = pd.Series({"A": 0.5, "B": 0.5})
        bw = pd.Series({"A": 0.5, "B": 0.5})
        pr = pd.Series({"A": 0.10, "B": 0.05})
        br = pd.Series({"A": 0.08, "B": 0.03})

        res = brinson_fachler(pw, bw, pr, br)
        assert res["total_allocation_effect"] == pytest.approx(0.0)
        assert res["total_interaction_effect"] == pytest.approx(0.0)
        assert res["total_selection_effect"] == pytest.approx(res["total_active_return"])


class TestSectorAttribution:
    def test_sector_weights_aggregate_correctly(self):
        pw = {"AAA": 0.3, "BBB": 0.2, "CCC": 0.5}
        pr = {"AAA": 0.05, "BBB": 0.04, "CCC": 0.02}
        sector_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}

        res = sector_attribution(
            pw, pr,
            benchmark_sector_weights={"Tech": 0.5, "Energy": 0.5},
            benchmark_sector_returns={"Tech": 0.03, "Energy": 0.02},
            sector_map=sector_map,
        )
        tech = next(s for s in res["sectors"] if s["sector"] == "Tech")
        energy = next(s for s in res["sectors"] if s["sector"] == "Energy")
        assert tech["port_weight"] == pytest.approx(0.5)
        assert energy["port_weight"] == pytest.approx(0.5)
        # Tech port return: (0.3*0.05 + 0.2*0.04)/0.5 = 0.046
        assert tech["port_return"] == pytest.approx(0.046)

    def test_components_sum_to_total(self):
        pw = {"AAA": 0.4, "BBB": 0.6}
        pr = {"AAA": 0.05, "BBB": 0.03}
        sector_map = {"AAA": "Tech", "BBB": "Energy"}

        res = sector_attribution(
            pw, pr,
            benchmark_sector_weights={"Tech": 0.6, "Energy": 0.4},
            benchmark_sector_returns={"Tech": 0.04, "Energy": 0.02},
            sector_map=sector_map,
        )
        summed = res["total_allocation"] + res["total_selection"] + res["total_interaction"]
        assert summed == pytest.approx(res["total_active"])

    def test_unknown_ticker_goes_to_other_bucket(self):
        pw = {"ZZZZ": 1.0}
        pr = {"ZZZZ": 0.1}
        res = sector_attribution(
            pw, pr,
            benchmark_sector_weights={"Other": 1.0},
            benchmark_sector_returns={"Other": 0.05},
            sector_map={},
        )
        sectors = {s["sector"] for s in res["sectors"]}
        assert "Other" in sectors

    def test_default_benchmark_weights_normalize_to_one(self):
        # No explicit benchmark provided: default SPY-like synthesis should
        # produce weights that sum to 1 across the portfolio's sectors.
        pw = {"TENB": 0.5, "LOAR": 0.5}
        pr = {"TENB": 0.05, "LOAR": 0.03}
        res = sector_attribution(pw, pr)
        total_bench_weight = sum(s["bench_weight"] for s in res["sectors"])
        assert total_bench_weight == pytest.approx(1.0)
