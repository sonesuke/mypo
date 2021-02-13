import numpy as np
import numpy.testing as npt
from my_portfolio import rebalance
from my_portfolio import calc_capital_gain_tax
from my_portfolio import calc_fee
from my_portfolio import calc_income_gain_tax


def test_rebalance():
    assets = np.array([1, 1])
    weights = np.array([0.6, 0.4])
    npt.assert_almost_equal(
        rebalance(assets, weights),
        np.array([0.2, -0.2])
        )


def test_capital_gain_tax():
    diff = np.array([0.1, -0.2])
    initial_assets = np.array([1, 1])
    assets = np.array([1.2, 1.1])
    npt.assert_almost_equal(
        calc_capital_gain_tax(initial_assets, assets, diff, 0.20),
        -0.004
        )


def test_income_gain_tax():
    price_dividends_yield = np.array([0.05, 0.01])
    assets = np.array([1.2, 1.1])
    npt.assert_almost_equal(
        calc_income_gain_tax(assets, price_dividends_yield, 0.20),
        -0.0142
    )


def test_fee():
    diff = np.array([0.1, -0.2])
    npt.assert_almost_equal(
        calc_fee(diff, 0.005),
        -0.0015
    )
