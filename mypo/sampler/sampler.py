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

    def __init__(self, market: Market, scenarios: int = 10) -> None:
        """Construct this object.

        Args:
            market: Market data.
            scenarios: Count of scenarios.
        """
        self.construct_model(market, scenarios)

    def save(self, filepath: str) -> None:
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

    def construct_model(self, market: Market, scenarios: int = 10) -> None:
        """Construct sampler model.

        Args:
            market: Market data.
            scenarios: Count of scenarios.
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
            low_triangular_cov = pm.LKJCholeskyCov("low_triangular_cov", n=n, eta=1, sd_dist=sd_dist)

            chol = pm.expand_packed_triangular(n, low_triangular_cov, lower=True)
            pm.MvNormal("observed_returns", mu=mu, chol=chol, observed=observed)

            trace = pm.sample(scenarios, pm.NUTS(), chains=3, return_inferencedata=False)
            self._mu = trace["mu"]
            self._chol = trace["low_triangular_cov"]
            self._columns = list(observed.columns)

    def sample(self, scenarios: int, samples: int, seed: int = 32) -> List[pd.DataFrame]:
        """Generate samples.

        Args:
            scenarios: Count of scenarios.
            samples: Count of samples in a scenario.
            seed: Seed for random.

        Returns:
            Scenarios.
        """
        ret = []
        count_of_mu, _ = self._mu.shape
        np.random.seed(seed=seed)
        for i in range(scenarios):
            mu = self._mu[i % count_of_mu]
            cov = Sampler._get_symmetric(self._chol[i % count_of_mu], len(self._columns))
            prices = multivariate_normal(mu, cov, samples)
            ret += [pd.DataFrame(prices, columns=self._columns)]
        return ret

    @staticmethod
    def _get_symmetric(elements: np.ndarray, n: int) -> np.ndarray:
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

        sym = np.array(element).reshape(2, 2)
        ret: np.ndarray = sym + sym.T - np.diag(sym.diagonal())
        return ret
