import os

from mypo import Loader

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_save_load():
    loader = Loader()
    loader.get(ticker="VOO", expense_ratio=0.0003)
    loader.get(ticker="IEF", expense_ratio=0.0015)
    market = loader.get_market()
    market.save(TEST_DATA)
    assert market is not None
