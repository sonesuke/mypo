import datetime
import os

import numpy.testing as npt

from mypo import Market
from mypo.optimizer import MinimumVarianceOptimizer, SharpRatioOptimizer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_minimum_variance_optimizer():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:300])
    optimizer = MinimumVarianceOptimizer(market)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.2501864, 0.7498136])


def test_minimum_variance_optimizer_with_minimum_return():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:300])
    optimizer = MinimumVarianceOptimizer(market=market)
    weights = optimizer.optimize_weight(minimum_return=0.08)
    npt.assert_almost_equal(weights, [0.1043978, 0.8956022])


def test_minimum_sharp_ratio_optimizer():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:300])
    optimizer = SharpRatioOptimizer(market)
    weights = optimizer.optimize_weight(risk_free_rate=0.02)
    npt.assert_almost_equal(weights, [1, 0])
