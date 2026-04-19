import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub yfinance if it isn't installed in the test environment. Tests that
# exercise code depending on yfinance.download should patch it explicitly.
if "yfinance" not in sys.modules:
    try:
        import yfinance  # noqa: F401
    except ImportError:
        stub = types.ModuleType("yfinance")
        stub.download = lambda *args, **kwargs: pd.DataFrame()
        sys.modules["yfinance"] = stub


@pytest.fixture
def daily_index():
    return pd.bdate_range("2022-01-03", periods=300)


@pytest.fixture
def normal_returns(daily_index):
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, len(daily_index)), index=daily_index)


@pytest.fixture
def benchmark_returns(daily_index):
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.0003, 0.009, len(daily_index)), index=daily_index)


@pytest.fixture
def constant_returns(daily_index):
    return pd.Series(0.001, index=daily_index)


@pytest.fixture
def zero_returns(daily_index):
    return pd.Series(0.0, index=daily_index)
