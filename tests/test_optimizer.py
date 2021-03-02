import datetime
import os

import numpy.testing as npt

from mypo import Market, MinimumVarianceOptimizer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_minimum_variance_optimizer():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:300])
    optimizer = MinimumVarianceOptimizer(market)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.2515872, 0.7484128])


def test_minimum_variance_optimizer_with_minimum_return():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:300])
    optimizer = MinimumVarianceOptimizer(market=market)
    weights = optimizer.optimize_weight(minimum_return=0.07)
    npt.assert_almost_equal(weights, [0.2007298, 0.7992702])
