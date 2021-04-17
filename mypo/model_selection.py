"""Utility functions for model selection."""

from __future__ import annotations

import itertools
from copy import deepcopy
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.vq import kmeans2
from scipy.optimize import lsq_linear
from tqdm import tqdm

from mypo.market import Market
from mypo.optimizer import BaseOptimizer


class Fold(object):

    _market: Market
    _train_span: int

    def __init__(self, market: Market, train_span: int):
        """Construct this object.

        Args:
            market: Market.
            train_span: Training span.
        """
        self._market = market
        self._train_span = train_span

    def get_train_span(self) -> int:
        """Get train span.

        Returns:
            Train span.
        """
        return self._train_span

    def get_train(self) -> Market:
        """Get train.

        Returns:
            Train.
        """
        return self._market.extract(self._market.get_index()[0 : self._train_span])

    def get_valid(self) -> Market:
        """Get validation.

        Returns:
            Validation.
        """
        return self._market

    def filter(self, tickers: List[str]) -> Fold:
        """Filter tickers.

        Args:
            tickers: Remaining tickers.

        Returns:
            Filtered Fold.
        """
        return Fold(market=self._market.filter(tickers), train_span=self._train_span)


def split_k_folds(market: Market, k: int, train_span: int) -> List[Fold]:
    """Split market to n periods.

    Args:
        market: Market data
        k: Count of split
        train_span: Train span if you want to specify.

    Returns:
        Split market data
    """
    index = market.get_index()
    len_index = len(index)

    eval_span = int((len_index - train_span) / k)
    folds: List[Fold] = []
    for i in range(k):
        start_eval_index = i * eval_span + train_span
        folds += [Fold(market.extract(index[start_eval_index - train_span : start_eval_index + eval_span]), train_span)]

    return folds


def clustering_tickers(market: Market, n: int, seed: int = 32) -> pd.DataFrame:
    """Enumerate tickers.

    Args:
        market: Market.
        n: Count of tickers.
        seed: Seed.

    Returns:
        Clustered tickers.
    """
    np.random.seed(seed)
    corr = market.get_rate_of_change().corr()

    _, label = kmeans2(corr.to_numpy(), k=n, minit="++")
    df = pd.DataFrame({"class": label}, index=corr.index)
    return df.sort_values("class")


def _make_combinations(cluster: pd.DataFrame) -> List[Tuple[str]]:
    """Make combinations.

    Args:
        cluster: Cluster data.

    Returns:
        Combinations.
    """
    k = cluster["class"].max() + 1
    clusters = []
    for i in range(k):
        clusters += [list(cluster[cluster["class"] == i].index)]
    return list(itertools.product(*clusters))


def evaluate_combinations(
    market: Market, cluster: pd.DataFrame, optimizer: BaseOptimizer, verbose: bool = False
) -> pd.DataFrame:
    """Evaluate combinations.

    Args:
        market: Market.
        cluster: Cluster data.
        verbose: Verbose mode.

    Returns:
        Result.
    """

    combinations = _make_combinations(cluster)

    def proc(c: Any) -> Any:  # pragma: no cover
        target_market = market.filter(list(c))
        optimized = optimizer.optimize(target_market, at=datetime.today())
        weights = optimizer.get_weights()
        return target_market.get_tickers(), optimized, weights

    def wrap(x: Any, total: Any) -> Any:
        """Wrapper for tqdm."""
        return tqdm(x, total=total) if verbose else x

    result = Parallel(n_jobs=-1)(delayed(proc)(c) for c in wrap(combinations, total=len(combinations)))

    df = pd.DataFrame(
        {
            "combinations": [r[0] for r in result],
            "optimized": [r[1] for r in result],
            "weights": [r[2] for r in result],
        }
    )
    return df.sort_values("optimized")


def select_by_correlation(market: Market, threshold: float) -> List[str]:
    """Select ticker by correlation.

    Args:
        market: Market.
        threshold: Threshold.

    Returns:
        Selected tickers.
    """
    df = market.get_rate_of_change().corr()
    corr = df.to_numpy()
    corr = np.tril(corr)
    corr = corr - np.diag(np.diag(corr))
    corr = np.where(np.abs(corr) > threshold, 1, 0)
    return list(df.columns[np.sum(corr, axis=1) > 0])


def select_by_regression(market: Market, threshold: float, verbose: bool = False) -> List[str]:
    """Select ticker by correlation.

    Args:
        market: Market.
        threshold: Threshold.

    Returns:
        Selected tickers.
    """
    tickers = select_by_correlation(market, threshold)
    df = market.get_rate_of_change()

    def wrap(x: Any) -> Any:
        """Wrapper for tqdm."""
        return tqdm(x) if verbose else x

    remains = deepcopy(tickers)
    for t in wrap(reversed(tickers)):
        remains.remove(t)
        A = df[remains].to_numpy()
        b = df[t]
        res = lsq_linear(A=A, b=b)
        corr = np.corrcoef(np.dot(A, res.x), b)[0, 1]
        if np.abs(corr) < threshold:
            remains.append(t)
    return remains
