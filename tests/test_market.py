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
    print(df)
    npt.assert_almost_equal(df["VOO"][0], 0.00453974)
    npt.assert_almost_equal(df["IEF"][0], -0.00288483)


def test_raw_price():
    market = Market.load(TEST_DATA)
    df = market.get_raw()
    print(df)
    npt.assert_almost_equal(df["VOO"][0], 82.4844284)
    npt.assert_almost_equal(df["IEF"][0], 79.2020874)


def test_normalized_price():
    market = Market.load(TEST_DATA)
    df = market.get_normalized_prices()
    print(df)
    npt.assert_almost_equal(df["VOO"][0], 1)
    npt.assert_almost_equal(df["IEF"][0], 1)


def test_dividends():
    market = Market.load(TEST_DATA)
    df = market.get_price_dividends_yield()
    npt.assert_almost_equal(df["VOO"][0], 0)
    npt.assert_almost_equal(df["IEF"][0], 0)
