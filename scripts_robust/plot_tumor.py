import os
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Euclidean",
    "Manhattan",
    #"Calibrated__Euclidean__mad",
    "Calibrated__Manhattan__mad",
    #"Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    #"Adaptive__Euclidean__cmad",
    #"Adaptive__Manhattan__cmad",
    #"Adaptive__Euclidean__mad_or_cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    "Info__Linear__Manhattan__mad_or_cmad",
]

data_dir = "data_robust"
n_rep = 1

for problem_type in [
    "tumor",
]:
    for i_rep in range(n_rep):
        for kwargs in [{'frac_error': 0}, {'frac_error': 0.2}, {'frac_error': 0.1}]:
            if problem_type == "tumor":
                problem = slad.TumorErrorProblem(**kwargs)

            slad.plot_cis(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}")
            plt.savefig(f"plot_robust/{problem.get_id()}_{i_rep}_cis.png")

            slad.plot_1d_kdes(problem, distance_names, data_dir=data_dir, problem_suff=f"_{i_rep}")
            plt.savefig(f"plot_robust/{problem.get_id()}_{i_rep}_1d_kdes.png")
