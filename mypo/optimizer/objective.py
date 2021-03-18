"""Covariance and semi-covariance."""
from typing import Callable

import numpy as np

CovarianceModel = Callable[[np.ndarray], np.ndarray]


def covariance(prices: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix.

    Parameters
    ----------
    prices
        Prices of market data.

    Returns
    -------
    Covariance matrix.

    """
    Q = np.cov(prices.T)
    return np.array(Q)


def semi_covariance(prices: np.ndarray) -> np.ndarray:
    """
    Calculate semi-covariance matrix.

    Parameters
    ----------
    prices
        Prices of market data.

    Returns
    -------
    Semi-covariance matrix.

    """
    Q = np.cov(np.where(prices.T < 0, prices.T, 0.0))
    return np.array(Q)


def sharp_ratio(r: np.float64, q: np.float64, risk_free_rate: np.float64) -> np.float64:
    """
    Calculate Sharp ratio.

    Parameters
    ----------
    r
        Return
    q
        Variance
    risk_free_rate
        Risk free rate

    Returns
    -------
    Sharp ratio.

    """
    return (r - risk_free_rate) / q
