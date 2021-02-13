import numpy as np
import numpy.testing as npt
from my_portfolio import Simulator


def test_simulator():
    simulator = Simulator(
        assets=np.array([1, 1]),
        weights=np.array([0.6, 0.4]),
        cash=0.5,
        spending=0.06
    )
    npt.assert_almost_equal(
        simulator.total_assets(),
        2.5
    )


def test_apply():
    simulator = Simulator(
        assets=np.array([1.2, 0.8]),
        weights=np.array([0.6, 0.4]),
        cash=0.5,
        spending=0.06
    )
    simulator.apply(
        market=np.array([1.1, 0.9]),
        price_dividends_yield=np.array([0.05, 0.01]),
        expense_ratio=np.array([0.0007, 0.0007])
    )
    npt.assert_almost_equal(
        simulator.total_assets(),
        2.625068
    )
