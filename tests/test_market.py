import os

import numpy.testing as npt
import pandas as pd

from mypo import Market

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_save_load():
    market = Market.load(TEST_DATA)
    index = market.get_rate_of_change().index
    assert index[0] == pd.Timestamp("2010-09-10")


def test_rate_of_change():
    market = Market.load(TEST_DATA)
    df = market.get_rate_of_change()
    npt.assert_almost_equal(df["VOO"][0], 0.00453974)
    npt.assert_almost_equal(df["IEF"][0], -0.00288483)


def test_raw_price():
    market = Market.load(TEST_DATA)
    df = market.get_raw()
    npt.assert_almost_equal(df["VOO"][0], 82.4844284)
    npt.assert_almost_equal(df["IEF"][0], 79.2020874)


def test_normalized_price():
    market = Market.load(TEST_DATA)
    df = market.get_normalized_prices()
    npt.assert_almost_equal(df["VOO"][0], 1)
    npt.assert_almost_equal(df["IEF"][0], 1)


def test_dividends():
    market = Market.load(TEST_DATA)
    df = market.get_price_dividends_yield()
    npt.assert_almost_equal(df["VOO"][0], 0)
    npt.assert_almost_equal(df["IEF"][0], 0)


def test_make_market():
    market = Market.create(
        ticker="NONE", start_date="2021-01-01", end_date="2021-12-31", yearly_gain=0.01
    )
    df = market.get_raw()
    assert df.index[0] == pd.Timestamp("2021-01-01")
    assert df.index[-1] == pd.Timestamp("2021-12-31")
    npt.assert_almost_equal(df["NONE"][0], 1)
    npt.assert_almost_equal(df["NONE"][-1], 1.01, decimal=3)

    df = market.get_price_dividends_yield()
    npt.assert_almost_equal(df["NONE"].sum(), 0)
