import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import scipy.stats as stats
import pickle
import argparse

import slad
from slad import C
from slad.util import *
from slad.ana_util import *
import pyabc

pyabc.settings.set_figure_params("pyabc")

data_dir = "data_learn_useall"
hist_suf = ""

n_rep = 10

fontsize_big = 12
fontsize_medium = 10
fontsize_small = 8
padding = 0.3

distance_names = [
    "Adaptive",
    # Linear
    "Linear__Initial",
    "Linear",
    "Adaptive__Linear",
    "Adaptive__Linear__Extend",
    # MLP
    "MLP2__Initial",
    "MLP2",
    "Adaptive__MLP2",
    "Adaptive__MLP2__Extend",
    # MS
    #"MS2__Initial",
    #"MS2",
    #"Adaptive__MS2",
    #"Adaptive__MS2__Extend",
    # Info Linear
    "Info__Linear__Initial",
    "Info__Linear",
    "Info__Linear__Extend",
    # Info MLP
    "Info__MLP2__Initial",
    "Info__MLP2",
    "Info__MLP2__Extend",
    # Info MS
    #"Info__MS2__Initial",
    #"Info__MS2",
    #"Info__MS2__Extend",
]

def create_vals(problem_type):
    problem = type_to_problem(problem_type)

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(distance_names)

    vals = np.full(shape=(n_dist, n_rep), fill_value=np.nan)

    for i_dist, distance_name in enumerate(distance_names):
        for i_rep in range(n_rep):
            h = pyabc.History(
                f"sqlite:///{data_dir}/{problem.get_id()}_{i_rep}/db_{distance_name}.db",
                create=False)
            _, w = h.get_distribution(t=h.max_t)
            vals[i_dist, i_rep] = pyabc.effective_sample_size(w)

    means = np.mean(vals, axis=1)
    stds = np.std(vals, axis=1)
    stds = np.minimum(means, stds)

    return means, stds, gt_par


def plot_ess(
    problem_type,
    log: bool,
    axes,
    ylabels: bool,
):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(distance_names)
    colors = [C.distance_colors_learn[dname] for dname in distance_names]
    for i_par in [0]:
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                slad.C.distance_labels_short_learn[dname] for dname in distance_names],
                fontdict={"fontsize": fontsize_medium},
            )
            ax.yaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()
        ax.barh(
            ys, means,
            xerr=stds,
            color=colors, alpha=0.9, height=0.7,
            error_kw={"ecolor": "grey"},
        )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            max_val = means.max()
            if log:
                pos_x = means[i_dist] * (1 + 1 / max_val)
            else:
                pos_x = means[i_dist] + (1  + 1 / max_val)
            mean = means[i_dist]
            if mean == 0:
                continue
            ax.text(max_val * 0.9,
                    i_dist,
                    f"{means[i_dist]:.1f}",
                    fontdict={"fontsize": fontsize_medium},
                    verticalalignment="center",
                    horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        #ax.set_title(slad.C.parameter_labels[problem_type][key], fontsize=fontsize_medium)
        ax.axhline(y=0.5, color="grey", linestyle="--")
        ax.axhline(y=4.5, color="grey", linestyle="--")
        ax.axhline(y=8.5, color="grey", linestyle="--")
        ax.axhline(y=11.5, color="grey", linestyle="--")

        plt.setp(ax.get_xticklabels(), fontsize=fontsize_small)
        plt.setp(ax.get_xminorticklabels(), visible=False)

    axes[0].text(
        0, 1.05, slad.C.problem_labels_learn[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=fontsize_big,
    )

problem_types = [
    "cr",
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
    #"harrison_toy",
]
arr_cols = [1] * len(problem_types)
fig, axes = plt.subplots(
    nrows=1, ncols=sum(arr_cols), figsize=(12, len(distance_names) / 2.5),
    #constrained_layout=True,
)

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_ess(
        problem_type=problem_type,
        log=False,
        axes=axes_for_problem,
        ylabels=i==0,
    )

# fig.tight_layout()
plt.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.1)

# x axis label
fig.text(
    0.5, 0.03, "Effective sample size",
    horizontalalignment="center", verticalalignment="center",
    fontsize=fontsize_medium,
)

#plt.subplots_adjust(bottom=0.1)

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_ess{hist_suf}.{fmt}", format=fmt, dpi=200)
