from typing import Callable
import numpy as np

import pyabc

from .base import Problem

try:
    import tumor2d
except ImportError:
    tumor2d = None


class TumorProblem(Problem):
    def __init__(self, obs_rep: int = 10):
        self.p_keys = [
            "growth_curve",
            "extra_cellular_matrix_profile",
            "proliferation_profile",
        ]

        refval = {
            "log_division_rate": 4.17e-2,
            "log_initial_spheroid_radius": 1.2e1,
            "log_initial_quiescent_cell_fraction": 7.5e-1,
            "log_division_depth": 100,
            "log_ecm_production_rate": 5e-3,
            "log_ecm_degradation_rate": 8e-4,
            "log_ecm_division_threshold": 1e-2,
        }
        for key, val in refval.items():
            refval[key] = np.log10(val)
        self.refval = refval

        self.limits = {
            "log_division_rate": (-3, -1),
            "log_division_depth": (1, 3),
            "log_initial_spheroid_radius": (0, 1.2),
            "log_initial_quiescent_cell_fraction": (-5, 0),
            "log_ecm_production_rate": (-5, 0),
            "log_ecm_degradation_rate": (-5, 0),
            "log_ecm_division_threshold": (-5, 0),
        }

        self.obs_rep: int = obs_rep

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(
            **{
                key: pyabc.RV("uniform", a, b - a)
                for key, (a, b) in self.limits.items()
            },
        )

    def get_prior_bounds(self) -> dict:
        return self.limits

    def get_obs(self) -> dict:
        model = self.get_model()
        datas = [model(self.refval) for _ in range(self.obs_rep)]
        data = {}
        for key in self.p_keys:
            data[key] = np.mean(np.array([d[key] for d in datas]), axis=0)
        return data

    def get_gt_par(self) -> dict:
        return self.refval

    def get_id(self) -> str:
        return "tumor2d"

    def get_model(self) -> Callable:
        def model(p: dict):
            ret = tumor2d.log_model(p)
            for key in ["extra_cellular_matrix_profile", "proliferation_profile"]:
                ret[key] = ret[key][::10]
            return ret

        return model
