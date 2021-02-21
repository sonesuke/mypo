from mypo import Loader
import pandas as pd
import numpy.testing as npt
import os

TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'test.bin')


def test_save_load():
    loader = Loader()
    loader.get('VOO')
    loader.get('IEF')
    loader.save(TEST_DATA)
    loader = Loader.load(TEST_DATA)
    index = loader.get_market().index
    assert index[0] == pd.Timestamp('2010-09-10')


def test_market():
    loader = Loader.load(TEST_DATA)
    df = loader.get_market()
    npt.assert_almost_equal(
        df['VOO'][0],
        1.0045397455037965
    )
    npt.assert_almost_equal(
        df['IEF'][0],
        0.9971151743025434
    )


def test_dividends():
    loader = Loader.load(TEST_DATA)
    df = loader.get_price_dividend_yield()
    npt.assert_almost_equal(
        df['VOO'][0],
        0
    )
    npt.assert_almost_equal(
        df['IEF'][0],
        0
    )
