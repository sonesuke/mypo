import numpy.testing as npt

from mypo import calc_capital_gain_tax, calc_fee, calc_income_gain_tax
from mypo.settings import DEFAULT_SETTINGS


def test_capital_gain_tax() -> None:
    diff = [0.1, -0.2]
    initial_assets = [1, 1]
    assets = [1.2, 1.1]
    npt.assert_almost_equal(calc_capital_gain_tax(initial_assets, assets, diff, DEFAULT_SETTINGS), -0.004)


def test_income_gain_tax() -> None:
    price_dividends_yield = [0.05, 0.01]
    assets = [1.2, 1.1]
    npt.assert_almost_equal(calc_income_gain_tax(assets, price_dividends_yield, DEFAULT_SETTINGS), -0.0142)


def test_fee() -> None:
    diff = [0.1, -0.2]
    npt.assert_almost_equal(calc_fee(diff, DEFAULT_SETTINGS), 0.0015)
