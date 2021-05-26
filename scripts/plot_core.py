import os
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

labels = [
    "Euclidean",
    "Calibrated",
    # for scale function comparison
    "Adaptive_std",
    "Adaptive",
    "Euclidean_Linear",
    "Linear_initial",
    "Linear",
    "Linear_Subset",
    "MS",
    "MS_Subset",
    # info distances
    "Info",
    "Info_Subset",
]

iters = list(range(10))
problem = slad.CoreProblem()

slad.plot_cis(problem, labels, iters)
plt.savefig(f"plot/{problem.get_id()}_cis.png")

slad.plot_1d_kdes(problem, labels, iters)
plt.savefig(f"plot/{problem.get_id()}_1d_kdes.png")
