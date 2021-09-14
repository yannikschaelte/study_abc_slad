"""Demonstration problem."""

import numpy as np
from pyabc import Distribution, RV
from typing import Any, Callable, Dict
from .base import Problem


class DemoProblem(Problem):
    """Demonstration problem.

    Summary statistics:
    * s1 is informative of parameter p1, both small-scale, with
      a relatively wider prior, such that calibration also does not work
    * s2 is informative of parameter p2, both large-scale
    * s3 is quadratic in parameter p3, inducing a bimodal posterior
    * s4 consists in multiple uninformative variables
    """

    def get_model(self) -> Callable:
        def model(p):
            vals = np.full(1+3+1+10, fill_value=np.nan)
            vals[0] = p["p1"] + 1e-1 * np.random.normal()
            vals[1:4] = p["p2"] + 1e2 * np.random.normal(size=3)
            vals[4] = p["p3"] ** 2 + 5e-2 * np.random.normal()
            vals[5:] = 1e1 * np.random.normal(size=10)
            return {"s": vals}

        return model

    def get_prior_bounds(self) -> dict:
        return {
            "p1": (-7e0, 7e0),
            "p2": (-7e2, 7e2),
            "p3": (-1e0, 1e0),
        }

    def get_prior(self) -> Distribution:
        return Distribution(
            **{
                key: RV("uniform", lb, ub - lb)
                for key, (lb, ub) in self.get_prior_bounds().items()
            },
        )

    def get_gt_par(self) -> dict:
        return {"p1": 0, "p2": 0, "p3": 0.7}

    def get_obs(self) -> dict:
        return {"s": np.array([0, *[0] * 3, 0.7**2, *[0] * 10])}
        # return self.get_model()(self.get_gt_par())

    def get_id(self) -> str:
        return "demo"


class OldDemoProblem(Problem):
    """Demonstration problem, a simplified version of the core problem.

    s0: informative of p0, small range, 1e-1, std 1e-2
    s1: informative of p1, large range, 1e3, std 1e0 -> rel. to prior denser
    s2: uninformative, large range
    s3: uninformative, small range
    """

    def __init__(
        self,
        n0: int = 1,
        n1: int = 1,
        n2: int = 10,
        std0: float = 0.1 * 1e-2,
        std1: float = 0.1 * 1e0,
        std2: float = 1e2,
        ub0: float = 1e-2,
        ub1: float = 1e2,
    ):
        self.n0, self.n1, self.n2 = n0, n1, n2
        self.std0, self.std1, self.std2 = std0, std1, std2
        self.ub0, self.ub1 = ub0, ub1

    def get_model(self):
        def model(p):
            return {
                "s0": p["p0"]
                + self.std0 * np.sqrt(self.n0) * np.random.normal(size=self.n0),
                "s1": p["p1"]
                + self.std1 * np.sqrt(self.n1) * np.random.normal(size=self.n1),
                "s2": 500
                + self.std2 * np.sqrt(self.n2) * np.random.normal(size=self.n2),
            }

        return model

    def get_prior(self) -> dict:
        return Distribution(
            p0=RV("uniform", 0, self.ub0),
            p1=RV("uniform", 0, self.ub1),
        )

    def get_prior_bounds(self) -> dict:
        gt_par = self.get_gt_par()
        s = 10
        return {"p0": (0, self.ub0), "p1": (0, self.ub1)}
        return {
            "p0": (
                max(gt_par["p0"] - s * self.std0, 0),
                min(gt_par["p0"] + s * self.std0, self.ub0),
            ),
            "p1": (
                max(gt_par["p1"] - s * self.std1, 0),
                min(gt_par["p1"] + s * self.std1, self.ub1),
            ),
        }
        # return {'p0': (0, self.ub0), 'p1': (0, self.ub1)}

    def get_gt_par(self) -> dict:
        return {
            "p0": 0.4 * self.ub0,
            "p1": 0.4 * self.ub1,
        }

    def get_obs(self) -> dict:
        gt_par = self.get_gt_par()
        return {
            "s0": gt_par["p0"] * np.ones(self.n0),
            "s1": gt_par["p1"] * np.ones(self.n1),
            "s2": 500 * np.ones(self.n2),
        }

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 50000}

    def get_id(self) -> str:
        return "core"
