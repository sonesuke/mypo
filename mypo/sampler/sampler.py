"""Sampler class for generating scenarios."""
from __future__ import annotations

import pickle
from typing import List

import numpy as np
import pandas as pd
from numpy.random import multivariate_normal

from mypo.market import Market


class Sampler(object):
    """Sampler class for generating scenarios."""

    _mu: np.ndarray
    _chol: np.ndarray
    _columns: List[str]

    def __init__(self, market: Market, samples: int = 10) -> None:
        """Construct this object.

        Args:
            market: Market data.
            samples: Count of samples.
        """
        self.construct_model(market, samples)  # pragma: no cover

    def save(self, filepath: str) -> None:  # pragma: no cover
        """Save sampler data to file.

        Args:
            filepath: Path to file for storing data.
        """
        with open(filepath, "wb") as bin_file:
            pickle.dump(self, bin_file)

    @staticmethod
    def load(filepath: str) -> Sampler:
        """Load sampler data from file.

        Args:
            filepath: Path to file for loading data.

        Returns:
            Market object
        """
        with open(filepath, "rb") as bin_file:
            value: Sampler = pickle.load(bin_file)
            return value

    def construct_model(self, market: Market, samples: int = 10) -> None:  # pragma: no cover
        """Construct sampler model.

        Args:
            market: Market data.
            samples: Count of scenarios.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pymc3 as pm

        observed = market.get_rate_of_change()
        n = len(observed.columns)
        with pm.Model():
            # average
            prior_mu = pm.Uniform("prior_mu", -1, 1, shape=n)
            mu = pm.Normal("mu", mu=prior_mu, sigma=1, shape=n)

            sd_dist = pm.HalfCauchy.dist(beta=2.5)
            lower_triangular_chol = pm.LKJCholeskyCov("lower_triangular_chol", n=n, eta=1, sd_dist=sd_dist)

            chol = pm.expand_packed_triangular(n, lower_triangular_chol, lower=True)
            pm.MvNormal("observed_returns", mu=mu, chol=chol, observed=observed)

            trace = pm.sample(samples, pm.NUTS(), chains=3, return_inferencedata=False, random_seed=32)
            self._mu = trace["mu"]
            self._chol = trace["lower_triangular_chol"]
            self._columns = list(observed.columns)

    def sample(self, samples: int, seed: int = 32) -> pd.DataFrame:
        """Generate samples.

        Args:
            samples: Count of samples in a scenario.
            seed: Seed for random.

        Returns:
            Scenarios.
        """
        count_of_mu, _ = self._mu.shape
        np.random.seed(seed=seed)
        series = np.random.randint(0, count_of_mu, samples)
        prices = np.zeros(shape=(samples, len(self._columns)))
        for i, seed in enumerate(series):
            mu = self._mu[seed]
            cov = Sampler._get_covariance(self._chol[seed], len(self._columns))
            prices[i] = multivariate_normal(mu, cov, 1)[0]
        return pd.DataFrame(prices, columns=self._columns)

    @staticmethod
    def _get_covariance(elements: np.ndarray, n: int) -> np.ndarray:
        """Get symmetric matrix.

        Args:
            elements: Elements of low triangular matrix.
            n: Dimension of symmetric matrix.

        Returns:
            Symmetric matrix.
        """
        element = []
        for i in range(len(elements)):
            element += [elements[i]]
            element += [0 for _ in range(i + 1, n)]

        lower = np.array(element).reshape(2, 2)
        ret: np.ndarray = np.dot(lower, lower.T)
        return ret
