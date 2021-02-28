import numpy.testing as npt
import pandas as pd
import numpy as np

from mypo import negative_total_return, max_drawdown, max_drawdown_span


def test_negative_total_return():
    report = pd.DataFrame(
        {
            "total_assets": [1.0, 1.25],
            "capital_gain": [1.0, 1.0],
            "income_gain": [1.0, 1.0],
            "cash": [1.0, 1.0],
            "deal": [1, 1],
            "fee": [0.1, 0.1],
            "capital_gain_tax": [0.1, 0.1],
            "income_gain_tax": [0.1, 0.1],
        }
    )
    npt.assert_almost_equal(negative_total_return(report), -1.25)


def test_max_drawdown():
    total_assets = [1.0, 0.9, 0.9]
    zeros = np.zeros(len(total_assets))
    report = pd.DataFrame(
        {
            "total_assets": total_assets,
            "capital_gain": zeros,
            "income_gain": zeros,
            "cash": zeros,
            "deal": zeros,
            "fee": zeros,
            "capital_gain_tax": zeros,
            "income_gain_tax": zeros,
        }
    )
    npt.assert_almost_equal(max_drawdown(report), 0.81)


def test_max_drawdown_span():
    total_assets = [1.0, 0.9, 0.9, 0.9, 1.25, 1.1, 0.9, 0.9]
    zeros = np.zeros(len(total_assets))
    report = pd.DataFrame(
        {
            "total_assets": total_assets,
            "capital_gain": zeros,
            "income_gain": zeros,
            "cash": zeros,
            "deal": zeros,
            "fee": zeros,
            "capital_gain_tax": zeros,
            "income_gain_tax": zeros,
        }
    )
    npt.assert_almost_equal(max_drawdown_span(report), 3)
