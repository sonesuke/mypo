import numpy as np
import numpy.testing as npt
from mypo import Rebalancer


def test_rebalance():
    assets = np.array([1, 1])
    rebalancer = Rebalancer(
        weights=np.array([0.6, 0.4])
    )
    npt.assert_almost_equal(
        rebalancer.apply(assets),
        np.array([0.2, -0.2])
        )
