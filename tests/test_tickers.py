import os

import pytest

from mypo import Market

skip_long_tests = pytest.mark.skipif(True, reason="This test takes long time.")
TEST_20050101_DATA = os.path.join(os.path.dirname(__file__), "data", "all_since_20050101.bin")


def test_sample() -> None:
    market = Market.load(TEST_20050101_DATA)
    assert market is not None
