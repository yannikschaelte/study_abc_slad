"""Models based on [#jiangwu2017]_.

.. [#jiangwu2017]
    Jiang, Bai, et al. "Learning summary statistic for approximate Bayesian
    computation via deep neural network."
    Statistica Sinica (2017): 1595-1618.
"""

from typing import Callable
import numpy as np

import pyabc

from .base import Problem


class JiangMovingProblem(Problem):
    """Moving average problem.

    This is similar to Jiang et al., but the prior is not chosen to make the
    parameters identifiable.
    """

    def __init__(self):
        self.p: int = 100
        self.q: int = 2

    def get_model(self) -> Callable:
        def model(p):
            theta = np.array([p[f"p{i+1}"] for i in range(self.q)])
            z = np.random.normal(size=self.p + self.q)
            x = np.zeros(shape=self.p)
            q = self.q
            for j in range(self.p):
                x[j] = z[j + q] + (theta * z[q + (j - q) : q + j][::-1]).sum()
            return {"x": x}

        return model

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(
            p1=pyabc.RV("uniform", -2, 4),
            p2=pyabc.RV("uniform", -1, 2),
        )

    def get_prior_bounds(self) -> dict:
        return {"p1": (-2, 2), "p2": (-1, 1)}
        pass

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {"p1": 0.6, "p2": 0.2}

    def get_id(self) -> str:
        return "jiang_moving"

    def get_sufficient_sumstat(self) -> Callable:
        def sumstat(y):
            x = y["x"]
            return {
                "y0": 1 / (self.p - 1) * (x[0 : self.p - 1] * x[1 : self.p]).sum(),
                "y1": 1 / (self.p - 2) * (x[0 : self.p - 2] * x[2 : self.p]).sum(),
            }

        return sumstat
