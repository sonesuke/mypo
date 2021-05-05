import os

import numpy.testing as npt
import pytest

from mypo import Market
from mypo.sampler import Sampler

skip_long_tests = pytest.mark.skipif(True, reason="This test takes long time.")
TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")

MODEL_DATA = os.path.join(os.path.dirname(__file__), "data", "sample.bin")


@skip_long_tests
def test_save_load() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(10)
    sampler = Sampler(market, samples=5)
    sampler.save(MODEL_DATA)


def test_sample() -> None:
    sampler = Sampler.load(MODEL_DATA)
    samples = sampler.sample(100)
    npt.assert_almost_equal(samples.mean(), [0.0034562, 0.001737])
