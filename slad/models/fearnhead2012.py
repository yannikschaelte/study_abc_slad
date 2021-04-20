"""Models from Fearnhead, Prangle 2012, Constructing Summary Statistics for
ABC.
"""

import numpy as np
from typing import Any, Callable, Dict

from pyabc import Distribution, RV
import pyabc
from .base import Problem, gk


class FearnheadGKProblem(Problem):
    def get_model(self) -> Callable:
        n_sample = 10000
        ixs = np.linspace(0, n_sample, 102, dtype=int)[1:-1]

        def model(p):
            A, B, g, k = [p[key] for key in ["A", "B", "g", "k"]]
            c = 0.8
            vals = gk(A=A, B=B, c=c, g=g, k=k, n=n_sample)
            ordered = np.sort(vals)
            subset = ordered[ixs]
            return {"y": subset}

        return model

    def get_prior(self) -> Distribution:
        return Distribution(
            A=RV("uniform", 0, 10),
            B=RV("uniform", 0, 10),
            g=RV("uniform", 0, 10),
            k=RV("uniform", 0, 10),
        )

    def get_prior_bounds(self) -> dict:
        return {key: (0, 10) for key in ["A", "B", "g", "k"]}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {"A": 3, "B": 1, "g": 1.5, "k": 0.5}

    def get_sumstat(self) -> pyabc.Sumstat:
        return pyabc.IdentitySumstat(
            trafos=[lambda x: x, lambda x: x ** 2, lambda x: x ** 3, lambda x: x ** 4],
        )

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 1e6}


class FearnheadLVProblem(Problem):
    def get_model(self) -> Callable:
        import ssa

        reactants = np.array([[1, 0], [1, 1], [0, 1]])
        products = np.array([[2, 0], [0, 2], [0, 0]])

        x0 = np.array([71, 79])

        t_max = 20
        # output = ssa.output.FullOutput()
        ts = np.arange(0, t_max, 0.1)
        output = ssa.output.ArrayOutput(ts=ts)
        sigma = np.exp(2.3)

        def model(p):
            k = np.array([p["p1"], p["p2"], p["p3"]])
            ssa_model = ssa.Model(
                reactants,
                products,
                x0=x0,
                k=k,
                t_max=t_max,
                max_reactions=int(1e5),
                output=output,
            )
            try:
                ret = ssa_model.simulate()
                # shape: (t, 2)
                sims = ret.list_xs[0]
            except ValueError:
                sims = np.empty((ts.size, 2))
                sims[:] = np.nan

            return {
                "y": sims,
            }

        return model

    def get_prior(self) -> Distribution:
        return Distribution(
            p1=RV("expon", scale=1 / 100),
            p2=RV("expon", scale=1 / 100),
            p3=RV("expon", scale=1 / 100),
        )

    def get_prior_bounds(self) -> dict:
        return {"p1": (0, 1), "p2": (0, 0.005), "p3": (0, 0.6)}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {"p1": 0.5, "p2": 0.0025, "p3": 0.3}

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 50000}
