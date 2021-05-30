import os
from datetime import datetime

import pytest

from mypo import Loader

skip_long_tests = pytest.mark.skipif(True, reason="This test takes long time.")
TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")
LOADER_DATA = os.path.join(os.path.dirname(__file__), "data", "loader.bin")


@skip_long_tests
def test_save_load() -> None:
    loader = Loader()
    loader.get(ticker="VOO", expense_ratio=0.0003)
    loader.get(ticker="IEF", expense_ratio=0.0015)
    loader.save(LOADER_DATA)
    loader.load(LOADER_DATA)
    market = loader.get_market()
    market.save(TEST_DATA)
    assert market is not None


@skip_long_tests
def test_nem() -> None:
    loader = Loader()
    loader.get(ticker="NEM", expense_ratio=0.00)
    loader.get(ticker="BHP", expense_ratio=0.00)
    market = loader.get_market()
    assert market is not None


@skip_long_tests
def test_constructor() -> None:
    loader = Loader()
    loader.get(ticker="VOO")
    print(loader.summary())
    assert len(loader.summary()) == 1
    loader = Loader()
    loader.get(ticker="IEF")
    print(loader.summary())
    assert len(loader.summary()) == 1


def test_since() -> None:
    loader = Loader.load(LOADER_DATA)
    loader = loader.since(since=datetime(2009, 12, 31))
    market = loader.get_market()
    assert market.get_tickers() == ["IEF"]


def test_volume_than() -> None:
    loader = Loader.load(LOADER_DATA)
    loader = loader.volume_than(4706801)
    market = loader.get_market()
    assert market.get_tickers() == ["IEF"]


def test_total_asset_than() -> None:
    loader = Loader.load(LOADER_DATA)
    loader = loader.total_assets_than(13733128193)
    market = loader.get_market()
    assert market.get_tickers() == ["VOO"]


def test_names() -> None:
    loader = Loader.load(LOADER_DATA)
    assert loader._names["VOO"] == "Vanguard S&P 500 ETF"


def test_smmary() -> None:
    loader = Loader.load(LOADER_DATA)
    summary = loader.summary()
    assert summary is not None
