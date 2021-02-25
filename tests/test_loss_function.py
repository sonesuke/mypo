import numpy.testing as npt
import pandas as pd

from mypo import negative_total_return


def test_negative_total_return():
    report = pd.DataFrame(
        {
            "capital_gain": [1.0, 1.0],
            "income_gain": [1.0, 1.0],
            "cash": [1.0, 1.0],
            "deal": [1, 1],
            "fee": [0.1, 0.1],
            "capital_gain_tax": [0.1, 0.1],
            "income_gain_tax": [0.1, 0.1],
        }
    )
    npt.assert_almost_equal(negative_total_return(report), -3.8)
