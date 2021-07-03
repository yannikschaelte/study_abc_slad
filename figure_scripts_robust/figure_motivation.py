import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

problem = slad.UninfErrorProblem()
i_rep = 0

base_dir = os.path.join("data_hist", f"{problem.get_id()}_{i_rep}")

hs = [
    pyabc.History("sqlite:///" + os.path.join(base_dir, f"db_{dname}.db"), create=False)
    for dname in distance_names
]

labels = [slad.C.distance_labels_short[dname] for dname in distance_names]
colors = [slad.C.distance_colors[dname] for dname in distance_names]

fig, ax = plt.subplots(figsize=(5, 3))

for h, label, color in zip(hs, labels, colors):
    pyabc.visualization.plot_histogram_1d(
        h,
        x="p0",
        xname=r"$\theta$",
        xmin=problem.get_prior_bounds()["p0"][0],
        xmax=problem.get_prior_bounds()["p0"][1],
        refval=problem.get_gt_par(),
        refval_color="grey",
        color=color,
        label=label,
    )

for fmt in ["png", "svg"]:
    plt.savefig(f"figures_robust/motivation.{fmt}", format=fmt)
