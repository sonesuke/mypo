import os

import numpy.testing as npt
import pytest

from mypo import Market
from mypo.optimizer import CVaROptimizer, MinimumVarianceOptimizer, SharpRatioOptimizer
from mypo.sampler import Sampler

skip_long_tests = pytest.mark.skipif(True, reason="This test takes long time.")
TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")
MODEL_DATA = os.path.join(os.path.dirname(__file__), "data", "sample.bin")


def test_minimum_variance_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(300)
    optimizer = MinimumVarianceOptimizer()
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.249759, 0.750241])


def test_minimum_variance_optimizer_with_minimum_return() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MinimumVarianceOptimizer(minimum_return=0.10)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.330534, 0.669466], decimal=5)


def test_minimum_sharp_ratio_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = SharpRatioOptimizer(risk_free_rate=0.02)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.4380607, 0.5619393], decimal=5)


def test_semi_minimum_variance_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MinimumVarianceOptimizer(with_semi_covariance=True)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.3239604, 0.6760396], decimal=5)


def test_cvar_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = CVaROptimizer(sampler=Sampler.load(MODEL_DATA))
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [1, 0], decimal=5)


@skip_long_tests
def test_cvar_optimizer_with_sampling() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = CVaROptimizer(scenarios=10)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0, 1], decimal=5)
    assert False
