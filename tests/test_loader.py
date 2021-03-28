import os

import pytest

from mypo import Loader

skip_long_tests = pytest.mark.skipif(True, reason="This test takes long time.")
TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


@skip_long_tests
def test_save_load() -> None:
    loader = Loader()
    loader.get(ticker="VOO", expense_ratio=0.0003)
    loader.get(ticker="IEF", expense_ratio=0.0015)
    market = loader.get_market()
    market.save(TEST_DATA)
    assert market is not None
