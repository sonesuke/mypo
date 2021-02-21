from mypo import Market
import os
import pandas as pd
import numpy.testing as npt


TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'test.bin')


def test_save_load():
    market = Market.load(TEST_DATA)
    index = market.get_prices().index
    assert index[0] == pd.Timestamp('2010-09-10')


def test_market():
    market = Market.load(TEST_DATA)
    df = market.get_prices()
    npt.assert_almost_equal(
        df['VOO'][0],
        1.0045397455037965
    )
    npt.assert_almost_equal(
        df['IEF'][0],
        0.9971151743025434
    )


def test_dividends():
    market = Market.load(TEST_DATA)
    df = market.get_price_dividends_yield()
    npt.assert_almost_equal(
        df['VOO'][0],
        0
    )
    npt.assert_almost_equal(
        df['IEF'][0],
        0
    )
