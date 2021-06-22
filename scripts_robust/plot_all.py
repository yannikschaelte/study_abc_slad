import os
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Euclidean",
    "Manhattan",
    "Calibrated__Euclidean__mad",
    "Calibrated__Manhattan__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Euclidean__cmad",
    "Adaptive__Manhattan__cmad",
    "Adaptive__Euclidean__mad_or_cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad",
    "Info__Linear__Manhattan__mad_or_cmad__All",
    "Info__Linear__Manhattan__mad_or_cmad__Subset",
]

data_dir = "data_robust"
n_rep = 10

for problem_type in [
    "gaussian",
    "gk",
    "lv",
]:
    for i_rep in range(n_rep):
        for kwargs in [{'n_obs_error': 0}, {}]:
            if problem_type == "gaussian":
                problem = slad.GaussianErrorProblem(**kwargs)
            elif problem_type == "gk":
                problem = slad.PrangleGKErrorProblem(**kwargs)
            elif problem_type == "lv":
                problem = slad.PrangleLVErrorProblem(**kwargs)

            slad.plot_cis(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}")
            plt.savefig(f"plot_robust/{problem.get_id()}_{i_rep}_cis.png")

            slad.plot_1d_kdes(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}")
            plt.savefig(f"plot_robust/{problem.get_id()}_{i_rep}_1d_kdes.png")
