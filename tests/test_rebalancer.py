from datetime import datetime

import numpy as np
import numpy.testing as npt

from mypo import Market
from mypo.optimizer import NoOptimizer
from mypo.rebalancer import MonthlyRebalancer, NoRebalancer, PlainRebalancer, ThresholdRebalancer


def test_rebalance() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = PlainRebalancer(weights=weights)
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)
    npt.assert_almost_equal(np.sum(diff), 0)


def test_rebalance_imbalance() -> None:
    assets = [1.3, 1]
    weights = [0.3, 0.7]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = PlainRebalancer(weights=weights)
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)
    npt.assert_almost_equal(np.sum(diff), 0)


def test_rebalance_monthly_fire() -> None:
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = MonthlyRebalancer(NoOptimizer(weights))
    diff = rebalancer.apply(datetime(2021, 3, 1, 0, 0), market, assets, np.float64(0.5))
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)


def test_rebalance_monthly_not_fire() -> None:
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = MonthlyRebalancer(NoOptimizer(weights), old_month=2)
    diff = rebalancer.apply(datetime(2021, 2, 1, 0, 0), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(np.abs(diff)), 0)


def test_rebalance_threshold_fire() -> None:
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = ThresholdRebalancer(NoOptimizer(weights))
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)


def test_rebalance_threshold_not_fire() -> None:
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = ThresholdRebalancer(NoOptimizer(weights), threshold=np.float64(0.5))
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(np.abs(diff)), 0)


def test_rebalance_no() -> None:
    assets = [1.5, 1]
    rebalancer = NoRebalancer()
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(np.abs(diff)), 0)


def test_rebalance_monthly_fire_with_re_optimize() -> None:
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-02", yearly_gain=0.00)
    rebalancer = MonthlyRebalancer(NoOptimizer(weights, do_re_optimize=True))
    diff = rebalancer.apply(datetime(2021, 3, 1, 0, 0), market, assets, np.float64(0.5))
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)
