import os

import numpy.testing as npt
import pandas as pd

from mypo import Market

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_save_load():
    market = Market.load(TEST_DATA)
    index = market.get_prices().index
    assert index[0] == pd.Timestamp("2010-09-10")


def test_market():
    market = Market.load(TEST_DATA)
    df = market.get_prices()
    print(df)
    npt.assert_almost_equal(df["VOO"][0], 0.00453974)
    npt.assert_almost_equal(df["IEF"][0], -0.00288483)


def test_dividends():
    market = Market.load(TEST_DATA)
    df = market.get_price_dividends_yield()
    npt.assert_almost_equal(df["VOO"][0], 0)
    npt.assert_almost_equal(df["IEF"][0], 0)
