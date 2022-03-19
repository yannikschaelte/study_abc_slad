"""Models based on [#fearnheadpra2012]_.

.. [#fearnheadpra2012]
    Fearnhead, Paul, and Dennis Prangle.
    "Constructing summary statistics for approximate Bayesian computation:
    semi‐automatic approximate Bayesian computation."
    Journal of the Royal Statistical Society: Series B
    (Statistical Methodology)
    74.3 (2012): 419-474.
"""

import numpy as np
from typing import Any, Callable, Dict

from pyabc import Distribution, RV
import pyabc
from .base import Problem
from .util import gk


class FearnheadGKProblem(Problem):
    """g-and-k distribution problem as used in Fearnhead et al.."""

    def __init__(self, n_sample: int = 10000, n_sumstat: int = 100):
        self.n_sample: int = n_sample
        self.n_sumstat: int = n_sumstat

    def get_model(self) -> Callable:
        if self.n_sample == self.n_sumstat:
            ixs = np.arange(self.n_sample)
        else:
            ixs = np.linspace(0, self.n_sample, self.n_sumstat + 2, dtype=int)[1:-1]

        def model(p):
            A, B, g, k = [p[key] for key in ["A", "B", "g", "k"]]
            c = 0.8
            vals = gk(A=A, B=B, c=c, g=g, k=k, n=self.n_sample)
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
        # return {"A": (2.8, 3.2), "B": (0.8, 1.2), "g": (0, 4), "k": (0, 2)}
        return {key: (0, 10) for key in ["A", "B", "g", "k"]}

    def get_viz_bounds(self) -> dict:
        return {"A": (2.5, 3.5), "B": (0.5, 1.5), "g": (1, 2), "k": (0, 1)}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {"A": 3, "B": 1, "g": 2, "k": 0.5}

    def get_sumstat(self) -> pyabc.Sumstat:
        return pyabc.IdentitySumstat(
            trafos=[lambda x: x, lambda x: x ** 2, lambda x: x ** 3, lambda x: x ** 4],
        )

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 1e6}

    def get_id(self) -> str:
        return f"fearnhead_gk_{self.n_sample}_{self.n_sumstat}"


class FearnheadLVProblem(Problem):
    """Lotka-Volterra problem similar to its use in Fearnhead et al.
    (the prior may be different).
    """

    def __init__(self, obs_rep: int = 100):
        self.obs_rep: int = obs_rep

    def get_model(self) -> Callable:
        import ssa

        reactants = np.array([[1, 0], [1, 1], [0, 1]])
        products = np.array([[2, 0], [0, 2], [0, 0]])

        x0 = np.array([71, 79])

        t_max = 20
        # output = ssa.output.FullOutput()
        ts = np.arange(0, t_max, 0.1)
        output = ssa.output.ArrayOutput(ts=ts)

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
                # shape: (n_t, 2)
                sims = ret.list_xs[0]
            except ValueError:
                sims = np.empty((ts.size, 2))
                # this value is certainly far away from typical simulations
                # nan would require special handling in the distance function
                sims[:] = 0
            return {
                "y": sims,
            }

        return model

    def get_prior(self) -> Distribution:
        return Distribution(
            p1=RV("uniform", 0, 2),
            p2=RV("uniform", 0, 0.1),
            p3=RV("uniform", 0, 1),
        )

    def get_prior_bounds(self) -> dict:
        return {"p1": (0, 2), "p2": (0, 0.1), "p3": (0, 1)}

    def get_viz_bounds(self) -> dict:
        return {"p1": (0, 1.5), "p2": (0, 0.02), "p3": (0, 1)}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())
        #model = self.get_model()
        #datas = [model(self.get_gt_par()) for _ in range(self.obs_rep)]
        #data = {}
        #for key in datas[0].keys():
        #    data[key] = np.mean(np.array([d[key] for d in datas]), axis=0)
        #return data

    def get_gt_par(self) -> dict:
        return {"p1": 0.5, "p2": 0.0025, "p3": 0.3}

    def get_ana_args(self) -> Dict[str, Any]:
        return {"population_size": 1000, "max_total_nr_simulations": 50000}

    def get_id(self) -> str:
        # return "fearnhead_lv"
        return "fearnhead_lv_500"
