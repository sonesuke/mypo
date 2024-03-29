from datetime import datetime

import numpy as np
import numpy.testing as npt

from mypo import Market
from mypo.evacuator import CalendarEvacuator, CovarianceEvacuator, FixedEvacuator, MovingAverageEvacuator
from mypo.rebalancer import PlainRebalancer


def test_moving_average_evacuator_fire() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-31", yearly_gain=0.00, n_assets=2)
    market._closes["None0"].iloc[-1] = 0.5
    rebalancer = PlainRebalancer(weights=weights, evacuator=MovingAverageEvacuator())
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(assets + diff), 0)


def test_moving_average_evacuator_no_fire() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-31", yearly_gain=0.00, n_assets=2)
    market._closes["None0"].iloc[-1] = 0.5
    rebalancer = PlainRebalancer(weights=weights, evacuator=MovingAverageEvacuator(span=200))
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0))
    npt.assert_almost_equal(np.sum(diff), 0)


def test_covariance_evacuator() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-31", yearly_gain=0.00, n_assets=2)
    market._closes["None0"].iloc[-1] = 0.5
    rebalancer = PlainRebalancer(weights=weights, evacuator=CovarianceEvacuator(long_span=3, short_span=2))
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(assets + diff), 2.1875)


def test_fixed_evacuator_ratio() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-31", yearly_gain=0.00, n_assets=2)
    market._closes["None0"].iloc[-1] = 0.5
    rebalancer = PlainRebalancer(weights=weights, evacuator=FixedEvacuator(ratio=0.1))
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(assets + diff), 2.25)


def test_fixed_evacuator_level() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-31", yearly_gain=0.00, n_assets=2)
    market._closes["None0"].iloc[-1] = 0.5
    rebalancer = PlainRebalancer(weights=weights, evacuator=FixedEvacuator(level=0.1))
    diff = rebalancer.apply(datetime.now(), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(assets + diff), 2.4)


def test_calender_evacuator() -> None:
    assets = [1, 1]
    weights = [0.6, 0.4]
    market = Market.create(start_date="2021-01-01", end_date="2021-01-31", yearly_gain=0.00, n_assets=2)
    market._closes["None0"].iloc[-1] = 0.5
    rebalancer = PlainRebalancer(weights=weights, evacuator=CalendarEvacuator(months=[1], risk_off=0.1))
    diff = rebalancer.apply(datetime(2021, 1, 1), market, assets, np.float64(0.5))
    npt.assert_almost_equal(np.sum(assets + diff), 0.25)
