import numpy as np


def rebalance(assets, weight):
    diff = weight * np.sum(assets) - assets
    return diff


def calc_tax(initial_assets, assets, diff, tax_rate):
    deal_sell = np.where(diff > 0, 0, diff)
    unrealized_gain = assets - initial_assets
    plus_unrealized_gain = np.where(unrealized_gain > 0, unrealized_gain, 0)
    return np.sum(plus_unrealized_gain * deal_sell * tax_rate)


def calc_fee(diff, fee_rate):
    deal = np.abs(diff)
    return -np.sum(deal * fee_rate)
