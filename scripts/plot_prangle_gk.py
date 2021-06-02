import os
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

labels = [
    "Euclidean",
    "Calibrated",
    # for scale function comparison
    "Adaptive",
    "Euclidean_Linear",
    "Adaptive_Linear_initial",
    "Adaptive_Linear",
    "Adaptive_MS",
    # info distances
    "Info_MS",
    "Info_Linear",
]

iters = list(range(3))
problem = slad.PrangleGKProblem()

slad.plot_cis(problem, labels, iters)
plt.savefig(f"plot/{problem.get_id()}_cis.png")

slad.plot_1d_kdes(problem, labels, iters)
plt.savefig(f"plot/{problem.get_id()}_1d_kdes.png")
