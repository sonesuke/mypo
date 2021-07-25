# flake8: noqa
__version__ = "0.0.0"  # refresh automatically by CI

from .common import calc_capital_gain_tax, calc_fee, calc_income_gain_tax, covariance, sharpe_ratio
from .loader import Loader
from .market import Market, SamplingMethod
from .model_selection import (
    Fold,
    clustering_tickers,
    evaluate_combinations,
    select_by_correlation,
    select_by_regression,
    split_k_folds,
)
from .reporter import Reporter
from .runner import Runner
from .settings import Settings
