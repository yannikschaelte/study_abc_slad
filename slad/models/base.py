from typing import Any, Callable, Dict
import numpy as np
from abc import ABC, abstractmethod
import pyabc
from pyabc import Distribution, RV


class Problem(ABC):
    @abstractmethod
    def get_model(self) -> Callable:
        """Get the model."""

    @abstractmethod
    def get_prior(self) -> pyabc.Distribution:
        """Get the prior."""

    @abstractmethod
    def get_prior_bounds(self) -> dict:
        """Get prior boundaries"""

    @abstractmethod
    def get_obs(self) -> dict:
        """Get the observation."""

    @abstractmethod
    def get_gt_par(self) -> dict:
        """Get the ground truth parameters."""

    def get_sumstat(self) -> pyabc.Sumstat:
        """Get summary statistic function."""
        return pyabc.IdentitySumstat()

    @abstractmethod
    def get_ana_args(self) -> Dict[str, Any]:
        """Get analysis arguments"""


def gk(A, B, c, g, k, n: int = 1):
    """One informative, one uninformative statistic"""
    z = np.random.normal(size=n)
    e = np.exp(-g * z)
    return A + B * (1 + c * (1 - e) / (1 + e)) * (1 + z ** 2) ** k * z


class CoreProblem(Problem):
    def __init__(
        self,
        n0: int = 1,
        n1: int = 1,
        n2: int = 1,
        n3: int = 1,
        std0: float = 1e-2,
        std1: float = 1e0,
        std2: float = 1e2,
        std3: float = 1e-4,
        ub0: float = 1e-1,
        ub1: float = 1e3,
    ):
        self.n0, self.n1, self.n2, self.n3 = n0, n1, n2, n3
        self.std0, self.std1, self.std2, self.std3 = std0, std1, std2, std3
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
                "s3": 0.5
                + self.std3 * np.sqrt(self.n3) * np.random.normal(size=self.n3),
                # "s4": 0,
            }

        return model

    def get_prior(self) -> dict:
        return Distribution(
            p0=RV("uniform", 0, self.ub0),
            p1=RV("uniform", 0, self.ub1),
            # p2=RV("uniform", 0, 1),
        )

    def get_prior_bounds(self) -> dict:
        gt_par = self.get_gt_par()
        s = 10
        return {
            "p0": (
                max(gt_par["p0"] - s * self.std0, 0),
                min(gt_par["p0"] + s * self.std0, self.ub0),
            ),
            "p1": (
                max(gt_par["p1"] - s * self.std1, 0),
                min(gt_par["p1"] + s * self.std1, self.ub1),
            ),
            # "p2": (0, 1),
        }
        # return {'p0': (0, self.ub0), 'p1': (0, self.ub1)}

    def get_gt_par(self) -> dict:
        return {
            "p0": 0.4 * self.ub0,
            "p1": 0.4 * self.ub1,
            # "p2": 0.5,
        }

    def get_obs(self) -> dict:
        gt_par = self.get_gt_par()
        return {
            "s0": gt_par["p0"],
            "s1": gt_par["p1"],
            "s2": 500,
            "s3": 0.5,
            # "s4": 0,
        }

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 50000}
