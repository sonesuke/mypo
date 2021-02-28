"""
Utility functions.

A collection of utlities by using in Runner class.

"""


from typing import Any

import numpy as np
import numpy.typing as npt


def safe_cast(value: Any) -> np.ndarray:
    """
    Cast array like type to numpy array.

    Parameters
    ----------
    value
        target value

    Returns
    -------
    casted_value
    """
    casted_value = np.array(list(value))
    return casted_value


def calc_capital_gain_tax(
    average_asset_prices: npt.ArrayLike,
    prices: npt.ArrayLike,
    diff: npt.ArrayLike,
    tax_rate: np.float64,
) -> np.float64:
    """
    Calculate capital gain from market data and assets.

    Parameters
    ----------
    average_asset_prices
        average assets from investment started.

    prices
        Current assets.

    diff
        Deal.

    tax_rate
        Ratio of tax for capital gain.

    Returns
    -------
    capital_gain_tax
    """
    average_asset_prices = safe_cast(average_asset_prices)
    prices = safe_cast(prices)
    diff = safe_cast(diff)
    deal_sell = np.where(diff > 0, 0, diff)
    unrealized_gain = prices - average_asset_prices
    plus_unrealized_gain = np.where(unrealized_gain > 0, unrealized_gain, 0)
    capital_gain_tax: np.float64 = np.sum(plus_unrealized_gain * deal_sell * tax_rate)
    return capital_gain_tax


def calc_income_gain_tax(
    assets: npt.ArrayLike, price_dividends_yield: npt.ArrayLike, tax_rate: np.float64
) -> np.float64:
    """
    Calculate income gain from market data and assets.

    Parameters
    ----------
    assets
        Current assets.

    price_dividends_yield
        Price dividends yield.

    tax_rate
        Ratio of tax for capital gain.

    Returns
    -------
    income_gain_tax
    """
    assets = safe_cast(assets)
    price_dividends_yield = safe_cast(price_dividends_yield)
    income_gain_tax: np.float64 = -np.sum(assets * price_dividends_yield * tax_rate)
    return income_gain_tax


def calc_fee(diff: npt.ArrayLike, fee_rate: np.float64) -> np.float64:
    """
    Calculate fee for dealing.

    Parameters
    ----------
    diff
        Deal.

    fee_rate
        Ratio of fee for trading.

    Returns
    -------
    fee
    """
    diff = safe_cast(diff)
    deal = np.abs(diff)
    fee: np.float64 = -np.sum(deal * fee_rate)
    return fee
