from typing import Any

import numpy as np
import numpy.typing as npt


def safe_cast(value: Any) -> np.ndarray:
    return np.array(list(value))


def calc_capital_gain_tax(
    initial_assets: npt.ArrayLike, assets: npt.ArrayLike, diff: npt.ArrayLike, tax_rate: np.float64
) -> np.float64:
    initial_assets = safe_cast(initial_assets)
    assets = safe_cast(assets)
    diff = safe_cast(diff)
    deal_sell = np.where(diff > 0, 0, diff)
    unrealized_gain = assets - initial_assets
    plus_unrealized_gain = np.where(unrealized_gain > 0, unrealized_gain, 0)
    return np.float64(np.sum(plus_unrealized_gain * deal_sell * tax_rate))


def calc_income_gain_tax(
    assets: npt.ArrayLike, price_dividends_yield: npt.ArrayLike, tax_rate: np.float64
) -> np.float64:
    assets = safe_cast(assets)
    price_dividends_yield = safe_cast(price_dividends_yield)
    return np.float64(-np.sum(assets * price_dividends_yield * tax_rate))


def calc_fee(diff: npt.ArrayLike, fee_rate: np.float64) -> np.float64:
    diff = safe_cast(diff)
    deal = np.abs(diff)
    return np.float64(-np.sum(deal * fee_rate))
