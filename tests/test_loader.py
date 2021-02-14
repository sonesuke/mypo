import numpy as np
import numpy.testing as npt
from my_portfolio import rebalance
from my_portfolio import calc_capital_gain_tax
from my_portfolio import calc_fee
from my_portfolio import calc_income_gain_tax
from my_portfolio import Loader


def test_loader():
    loader = Loader()
    loader.get('VOO')
    df = loader.get_market()
    print(df)
    assert False


def test_loader_2():
    loader = Loader()
    loader.get('VOO')
    loader.get('IEF')
    df = loader.get_market()
    print(df)
    assert False