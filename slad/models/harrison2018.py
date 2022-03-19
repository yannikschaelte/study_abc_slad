"""Models based on [#harrisonbak2020]_.

.. [#harrisonbak2020]
    Harrison, Jonathan U., and Ruth E. Baker.
    "An automatic adaptive method to combine summary statistics in approximate
    Bayesian computation."
    PloS one 15.8 (2020): e0236954.
"""

from typing import Callable
import numpy as np
import ssa

import pyabc

from .base import Problem


class HarrisonToyProblem(Problem):
    """Gaussian toy problem with estimation of upper bound."""

    def get_model(self) -> Callable:
        def model(p):
            theta = 10.0 ** p["theta"]
            return {"y": np.sort(np.random.uniform(0, theta, size=10))}

        return model

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(theta=pyabc.RV("uniform", 0, 2))

    def get_prior_bounds(self) -> dict:
        return {"theta": (0, 2)}

    def get_viz_bounds(self) -> dict:
        return {"theta": (0.5, 1.5)}

    def get_gt_par(self) -> dict:
        return {"theta": 1}

    def get_id(self) -> str:
        return "harrison_toy"


class HarrisonBimodalProblem(Problem):
    def get_model(self) -> Callable:
        def model(p):
            theta = np.array([p["p1"], p["p2"]])
            y = np.sin(theta) + 0.1 * np.random.normal(size=2)
            return {"y1": y, "y2": p["p3"] + 0.1 * np.random.normal()}

        return model

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(
            **{key: pyabc.RV("uniform", 0, 2 * np.pi) for key in ["p1", "p2", "p3"]}
        )

    def get_prior_bounds(self) -> dict:
        return {key: (0, 2 * np.pi) for key in ["p1", "p2", "p3"]}

    def get_gt_par(self) -> dict:
        return {"p1": np.pi, "p2": np.pi, "p3": np.pi}

    def get_id(self) -> str:
        return "harrison_bimodal"

    def get_obs(self) -> dict:
        return {"y1": np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2]), "y2": np.pi}


class HarrisonDeathProblem(Problem):
    """MJP death process with additional independent Gaussian term."""

    def get_model(self) -> Callable:
        reactants = np.array([[1]])
        products = np.array([[0]])

        x0 = np.array([10])

        t_max = 20
        n_t = 33
        ts = np.linspace(0, t_max, n_t)
        output = ssa.output.ArrayOutput(ts=ts)

        def model(p):
            k = 10.0 ** np.array([p["k"]])
            ssa_model = ssa.Model(
                reactants,
                products,
                x0=x0,
                k=k,
                t_max=t_max,
                output=output,
            )
            ret = ssa_model.simulate()
            sims = ret.list_xs[0]
            y = np.zeros(shape=len(sims) + 1)
            y[:-1] = sims.flatten()
            y[-1:] = 10.0 ** p["sigma"] * np.random.normal(size=1)
            return {
                "y": y,
            }

        return model

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(
            **{key: pyabc.RV("uniform", -3, 6) for key in ["k", "sigma"]}
        )

    def get_prior_bounds(self) -> dict:
        return {key: (-3, 3) for key in ["k", "sigma"]}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {"k": -1, "sigma": -2}

    def get_id(self) -> str:
        return "harrison_death"


class HarrisonDimerProblem(Problem):
    """MJP dimerization problem."""

    def get_model(self) -> Callable:
        reactants = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [2, 0, 0],
                [0, 1, 0],
            ]
        )
        products = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [2, 0, 0],
            ]
        )

        x0 = np.array([1e5, 0, 0])

        t_max = 100
        n_t = 8
        ts = np.geomspace(1, t_max, num=n_t)
        output = ssa.output.ArrayOutput(ts=ts)

        def model(p):
            k = np.array([p[f"k{i+1}"] for i in range(4)])
            k = 10.0 ** k
            ssa_model = ssa.Model(
                reactants,
                products,
                x0=x0,
                k=k,
                t_max=t_max,
                output=output,
            )
            ret = ssa_model.simulate()
            sims = ret.list_xs[0]
            return {
                "s1": sims[:, 0],
                "s2": sims[:, 1],
                "s3": sims[:, 2],
            }

        return model

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(
            k1=pyabc.RV("uniform", -2, 4),
            k2=pyabc.RV("uniform", -3, 4),
            k3=pyabc.RV("uniform", -5, 4),
            k4=pyabc.RV("uniform", -3, 4),
        )

    def get_prior_bounds(self) -> dict:
        return {
            "k1": (-2, 2),
            "k2": (-3, 1),
            "k3": (-5, -1),
            "k4": (-3, 1),
        }

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        k1, k2, k3, k4 = np.log10([1, 0.04, 0.002, 0.5])
        return {"k1": k1, "k2": k2, "k3": k3, "k4": k4}

    def get_id(self) -> str:
        return "harrison_dimer"
