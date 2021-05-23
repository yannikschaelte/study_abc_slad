from typing import Callable
import numpy as np
from pyabc import Distribution, RV
from .base import Problem


class MultiModalProblem(Problem):
    """
    One parameter.

    Notes
    -----
    0.01, 0.1, 1e-4, 1e2 with size=1 -> only GP works
    _, _, 0, 0 -> subsetting works
    """

    def get_model(self) -> Callable:
        def model(p):
            return {
                "s0": p["p0"] ** 2 + 0.01 * np.random.normal(),
                "s1": p["p1"] + 0.1 * np.random.normal(size=10),
                "s2": 1e-4 * np.random.normal(size=1),
                "s3": 1e2 * np.random.normal(size=1),
            }

        return model

    def get_prior_bounds(self) -> dict:
        return {"p0": (-5, 5), "p1": (-5, 5)}

    def get_prior(self) -> Distribution:
        return Distribution(
            **{
                key: RV("uniform", lb, ub - lb)
                for key, (lb, ub) in self.get_prior_bounds().items()
            },
        )

    def get_gt_par(self) -> dict:
        return {"p0": 1, "p1": 1}

    def get_obs(self) -> dict:
        return {
            "s0": 1,
            "s1": 1 * np.ones(shape=10),
            "s2": np.ones(shape=1),
            "s3": np.ones(shape=1),
        }

    def get_id(self) -> str:
        return "multi_modal"
