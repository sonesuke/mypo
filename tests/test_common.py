import numpy.testing as npt

from mypo import calc_capital_gain_tax, calc_fee, calc_income_gain_tax


def test_capital_gain_tax():
    diff = [0.1, -0.2]
    initial_assets = [1, 1]
    assets = [1.2, 1.1]
    npt.assert_almost_equal(
        calc_capital_gain_tax(initial_assets, assets, diff, 0.20), -0.004
    )


def test_income_gain_tax():
    price_dividends_yield = [0.05, 0.01]
    assets = [1.2, 1.1]
    npt.assert_almost_equal(
        calc_income_gain_tax(assets, price_dividends_yield, 0.20), -0.0142
    )


def test_fee():
    diff = [0.1, -0.2]
    npt.assert_almost_equal(calc_fee(diff, 0.005), -0.0015)
