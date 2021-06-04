from typing import Callable
import numpy as np

import pyabc

from .prangle2015 import *
from .tumor import *


class GaussianErrorProblem(Problem):
    def __init__(self, n_obs: int = 10, n_obs_error: int = 2):
        self.n_obs: int = n_obs
        self.n_obs_error: int = n_obs_error

    def get_prior(self) -> pyabc.Distribution:
        return pyabc.Distribution(p0=pyabc.RV("uniform", 0, 10))

    def get_prior_bounds(self) -> dict:
        return {"p0": (0, 10)}

    def get_obs(self) -> dict:
        model = self.get_model()
        obs = model(self.get_gt_par())
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        obs["y"][: self.n_obs_error] = 0
        return obs

    def get_gt_par(self) -> dict:
        return {"p0": 6.0}

    def get_id(self) -> str:
        return f"gaussian_error_{self.n_obs}_{self.n_obs_error}"

    def get_model(self) -> Callable:
        def model(p: dict):
            return {"y": p["p0"] + 0.2 * np.random.normal(size=self.n_obs)}

        return model


class PrangleGKErrorProblem(PrangleGKProblem):
    def __init__(self, n_obs_error: int = 1):
        self.n_obs_error: int = n_obs_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error > 0:
            err_ixs = np.random.permutation(len(obs["y"]))[: self.n_obs_error]
            obs["y"][err_ixs] = 0
        return obs

    def get_id(self) -> str:
        return f"prangle_gk_{self.n_obs_error}"


class PrangleLVErrorProblem(PrangleLVProblem):
    def __init__(self, n_obs_error: int = 3):
        self.n_obs_error: int = n_obs_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.n_obs_error > 0:
            err_ixs = np.random.permutation(len(obs["y"][:, 0]))[: self.n_obs_error]
            obs["y"][err_ixs, :] = 0
        return obs

    def get_id(self) -> str:
        return f"prangle_lv_{self.n_obs_error}"


class TumorErrorProblem(TumorProblem):
    def __init__(self, obs_rep: int = 1, frac_error: float = 0.1):
        super().__init__(obs_rep=obs_rep)
        self.frac_error: float = frac_error

    def get_obs(self) -> dict:
        obs = super().get_obs()
        obs = self.errorfy(obs)
        return obs

    def errorfy(self, obs: dict) -> dict:
        if self.frac_error > 0:
            for key in self.refval.keys():
                n_obs = len(obs[key])
                n_err = min(int(self.frac_error * n_obs), n_obs)
                err_ixs = np.random.permutation(n_obs)[:n_err]
                while any(np.isclose(obs[key][err_ixs], 0)):
                    err_ixs = np.random.permutation(n_obs)[:n_err]
                obs[key][err_ixs] = 0
        return obs

    def get_id(self) -> str:
        return f"tumor2d_{self.frac_error}"
