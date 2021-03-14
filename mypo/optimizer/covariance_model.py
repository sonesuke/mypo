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
    return np.array(Q / np.max(np.abs(Q)))


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
    return np.array(Q / np.max(np.abs(Q)))
