"""Utility functions.

A collection of utlities by using in Runner class.

"""


from typing import Any

import numpy as np
import numpy.typing as npt

from mypo.settings import Settings


def safe_cast(value: Any) -> np.ndarray:
    """Cast array like type to numpy array.

    Args:
        value: target value

    Returns:
        casted_value
    """
    return np.array(list(value), dtype=np.float64)


def calc_capital_gain_tax(
    average_asset_prices: npt.ArrayLike, prices: npt.ArrayLike, diff: npt.ArrayLike, settings: Settings
) -> np.float64:
    """Calculate capital gain from market data and assets.

    Args:
        average_asset_prices: average assets from investment started.
        prices: Current assets.
        diff: Deal.
        settings: Ratio of tax for capital gain.

    Returns:
        capital_gain_tax
    """
    average_asset_prices = safe_cast(average_asset_prices)
    prices = safe_cast(prices)
    diff = safe_cast(diff)
    deal_sell = np.where(diff > 0, 0, diff)
    unrealized_gain = prices - average_asset_prices
    plus_unrealized_gain = np.where(unrealized_gain > 0, unrealized_gain, 0)
    capital_gain_tax: np.float64 = -np.sum(plus_unrealized_gain * deal_sell * settings.tax_rate)
    return capital_gain_tax


def calc_income_gain_tax(assets: npt.ArrayLike, price_dividends_yield: npt.ArrayLike, settings: Settings) -> np.float64:
    """Calculate income gain from market data and assets.

    Args:
        assets: Current assets.
        price_dividends_yield: Price dividends yield.
        settings: Ratio of tax for capital gain.

    Returns:
        income_gain_tax
    """
    assets = safe_cast(assets)
    price_dividends_yield = safe_cast(price_dividends_yield)
    income_gain_tax: np.float64 = np.sum(assets * price_dividends_yield * settings.tax_rate)
    return income_gain_tax


def calc_fee(diff: npt.ArrayLike, settings: Settings) -> np.float64:
    """Calculate fee for dealing.

    Args:
        diff: Deal.
        settings: Ratio of fee for trading.

    Returns:
        fee
    """
    diff = safe_cast(diff)
    deal = np.abs(diff)
    fee: np.float64 = np.sum(deal * settings.fee_rate)
    return fee


def sharpe_ratio(prices: np.ndarray, risk_free_rate: np.float64 = np.float64(0.02)) -> np.float64:
    """Calculate Sharpe ratio.

    Args:
        prices: Prices
        risk_free_rate: Risk free rate

    Returns:
        Sharpe ratio.
    """
    daily_r = np.prod((1 + prices)) ** (1 / len(prices)) - 1.0
    daily_q = np.sqrt(np.sum((prices - daily_r) ** 2) / len(prices))

    yearly_r = (1 + daily_r) ** 252 - 1.0
    yearly_q = daily_q * np.sqrt(252)

    return np.float64((yearly_r - risk_free_rate) / yearly_q)


def covariance(prices: np.ndarray) -> np.ndarray:
    """Calculate covariance matrix.

    Args:
        prices: Prices of market data.

    Returns:
        Covariance matrix.
    """
    Q = np.cov(prices.T, ddof=0)
    return np.array(Q)


def semi_covariance(prices: np.ndarray) -> np.ndarray:
    """Calculate semi-covariance matrix.

    Args:
        prices: Prices of market data.

    Returns:
        Semi-covariance matrix.
    """
    Q = np.cov(np.where(prices.T < 0, prices.T, 0.0), ddof=0)
    return np.array(Q)
