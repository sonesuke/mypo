from datetime import date

import numpy as np
import numpy.testing as npt

from mypo import MonthlyRebalancer, PlainRebalancer, ThresholdRebalancer


def test_rebalance():
    assets = [1, 1]
    weights = [0.6, 0.4]
    rebalancer = PlainRebalancer(weights=weights)
    diff = rebalancer.apply(date.today(), assets, 0.5)
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)
    npt.assert_almost_equal(np.sum(diff), 0)


def test_rebalance_imbalance():
    assets = np.array([1.3, 1])
    weights = np.array([0.3, 0.7])
    rebalancer = PlainRebalancer(weights=weights)
    diff = rebalancer.apply(date.today(), assets, 0.5)
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)
    npt.assert_almost_equal(np.sum(diff), 0)


def test_rebalance_monthly_fire():
    assets = np.array([1.5, 1])
    weights = np.array([0.3, 0.7])
    rebalancer = MonthlyRebalancer(weights=weights)
    diff = rebalancer.apply(date(2021, 2, 15), assets, 0.5)
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)


def test_rebalance_monthly_not_fire():
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    rebalancer = MonthlyRebalancer(weights=weights, old_month=2)
    diff = rebalancer.apply(date(2021, 2, 15), assets, 0.5)
    npt.assert_almost_equal(np.sum(np.abs(diff)), 0)


def test_rebalance_threshold_fire():
    assets = np.array([1.5, 1])
    weights = np.array([0.3, 0.7])
    rebalancer = ThresholdRebalancer(weights=weights)
    diff = rebalancer.apply(date(2021, 2, 15), assets, 0.5)
    npt.assert_almost_equal((assets + diff) / np.sum(assets + diff), weights)


def test_rebalance_threshold_not_fire():
    assets = [1.5, 1]
    weights = [0.3, 0.7]
    rebalancer = ThresholdRebalancer(weights=weights, threshold=0.5)
    diff = rebalancer.apply(date(2021, 2, 15), assets, 0.5)
    npt.assert_almost_equal(np.sum(np.abs(diff)), 0)
