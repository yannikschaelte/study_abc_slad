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
    "Info__Linear__Manhattan__mad_or_cmad",
]

data_dir = "data_robust"

for problem_type in [
    #"gaussian",
    #"gk",
    "lv",
]:
    if problem_type == "gaussian":
        problem = slad.GaussianErrorProblem(n_obs_error=0)
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem(n_obs_error=0)
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem(n_obs_error=0)

    slad.plot_cis(problem, distance_names, data_dir=data_dir)
    plt.savefig(f"plot_robust/{problem.get_id()}_cis.png")

    slad.plot_1d_kdes(problem, distance_names, data_dir=data_dir)
    plt.savefig(f"plot_robust/{problem.get_id()}_1d_kdes.png")

    if problem_type == "gaussian":
        problem = slad.GaussianErrorProblem()
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem()
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem()

    slad.plot_cis(problem, distance_names, data_dir=data_dir)
    plt.savefig(f"plot_robust/{problem.get_id()}_cis.png")

    slad.plot_1d_kdes(problem, distance_names, data_dir=data_dir)
    plt.savefig(f"plot_robust/{problem.get_id()}_1d_kdes.png")
