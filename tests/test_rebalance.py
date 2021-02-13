import numpy as np
import numpy.testing as npt
from my_portfolio import rebalance, calc_tax, calc_fee


def test_rebalance():
    assets = np.array([1, 1])
    weights = np.array([0.6, 0.4])
    npt.assert_almost_equal(
        rebalance(assets, weights),
        np.array([0.2, -0.2])
        )


def test_tax():
    diff = np.array([0.1, -0.2])
    inital_assets = np.array([1, 1])
    assets = np.array([1.2, 1.1])
    npt.assert_almost_equal(
        calc_tax(inital_assets, assets, diff, 0.20),
        -0.004
        )


def test_fee():
    diff = np.array([0.1, -0.2])
    npt.assert_almost_equal(
        calc_fee(diff, 0.005),
        -0.0015
    )
