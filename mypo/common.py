import numpy as np


def safe_cast(value):
    return np.array(list(value))


def calc_capital_gain_tax(initial_assets, assets, diff, tax_rate):
    initial_assets = safe_cast(initial_assets)
    assets = safe_cast(assets)
    diff = safe_cast(diff)
    deal_sell = np.where(diff > 0, 0, diff)
    unrealized_gain = assets - initial_assets
    plus_unrealized_gain = np.where(unrealized_gain > 0, unrealized_gain, 0)
    return np.sum(plus_unrealized_gain * deal_sell * tax_rate)


def calc_income_gain_tax(assets, price_dividends_yield, tax_rate):
    assets = safe_cast(assets)
    price_dividends_yield = safe_cast(price_dividends_yield)
    return -np.sum(assets * price_dividends_yield * tax_rate)


def calc_fee(diff, fee_rate):
    diff = safe_cast(diff)
    deal = np.abs(diff)
    return -np.sum(deal * fee_rate)
