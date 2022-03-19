import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
from slad import C
from slad.util import *
from slad.ana_util import *
import pyabc

pyabc.settings.set_figure_params("pyabc")


n_rep = 10

fontsize = 10
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

problem_types = [
    "cr",
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
    #"harrison_toy",
]

def create_vals(problem_type):
    pickle_file = f"figures_learn/data_{problem_type}.pickle"
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            means, stds, gt_par = pickle.load(f)
            return means, stds, gt_par

    problem = type_to_problem(problem_type)

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(distance_names)

    vals = np.full(shape=(n_dist, n_par, n_rep), fill_value=np.nan)

    for i_dist, distance_name in enumerate(distance_names):
        for i_rep in range(n_rep):
            h = pyabc.History(
                f"sqlite:///data_learn_useall/{problem.get_id()}_{i_rep}/db_{distance_name}.db",
                create=False)

            df, w = h.get_distribution(t=h.max_t)
            vals[i_dist, :, i_rep] = np.array(
                [pyabc.weighted_mse(df[key], w, gt_par[key])
                 for key in gt_par])

            # print(vals[i_dist, i_mode, :, i_rep])

    means = np.mean(vals, axis=2)
    stds = np.std(vals, axis=2)

    # median as a more robust measure because values are partly heavily skewed
    means = np.median(vals, axis=2)
    stds = np.full_like(means, fill_value=np.nan)
    for i_dist in range(means.shape[0]):
        for i_par in range(means.shape[1]):
            stds[i_dist, i_par] = np.median(np.abs(vals[i_dist, i_par, :] - np.median(vals[i_dist, i_par, :])))

    with open(pickle_file, "wb") as f:
        pickle.dump((means, stds, gt_par), f)

    return means, stds, gt_par


def short_exp_s(val):
    s = f"{val:.1e}"
    return s
    s1, s2 = s.split("e")
    if s2[1] == "0":
        s2 = s2[0] + s2[2:]
    return s1 + "e" + s2


def plot_rmse(problem_type, log: bool, axes, ylabels: bool):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(distance_names)
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                C.distance_labels_short_learn[dname] for dname in distance_names],
                fontdict={"fontsize": fontsize_small})
            #ax.xaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()
        ax.barh(
            ys, means[:, i_par],
            xerr=stds[:, i_par],
            color=[C.distance_colors_learn[dname] for dname in distance_names], alpha=0.9, height=0.7,
            error_kw={"ecolor": "grey"},
        )
        if log:
            xmin, xmax = ax.get_xlim()
            #if xmax / xmin > 2:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            max_val = means[:, i_par].max()
            if log:
                pos_x = means[i_dist, i_par] * (1 + 1 / max_val)
            else:
                pos_x = means[i_dist, i_par] + (1  + 1 / max_val)
            ax.text(max_val * 0.99,
                    i_dist,
                    short_exp_s(means[i_dist, i_par]),
                    fontdict={"fontsize": fontsize_small},
                    verticalalignment="center",#"bottom" if i == 0 else "top",
                    horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        ax.set_title(C.parameter_labels[problem_type][key], fontsize=fontsize)
        ax.axhline(y=0.5, color="grey", linestyle="--")
        ax.axhline(y=4.5, color="grey", linestyle="--")
        ax.axhline(y=8.5, color="grey", linestyle="--")
        ax.axhline(y=11.5, color="grey", linestyle="--")
        #ax.axhline(y=15.5, color="grey", linestyle="--")
        #ax.axhline(y=18.5, color="grey", linestyle="--")

        plt.setp(ax.get_xticklabels(), fontsize=6)
        plt.setp(ax.get_xminorticklabels(), visible=False)
        plt.setp(ax.get_xmajorticklabels(), visible=True)

    # fig.suptitle(problem_labels[problem_type])
    axes[0].text(
        0, 1.1, C.problem_labels_learn[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=12)

arr_cols = [2, 1, 4, 3, 4, 3]#, 1]
fig, axes = plt.subplots(
    nrows=1, ncols=sum(arr_cols),
    figsize=(14, len(distance_names) / 2.5),
    #constrained_layout=True,
)

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_rmse(
        problem_type, log=True, axes=axes_for_problem, ylabels=i==0,
    )

#fig.subplots_adjust(hspace=0.01)
#fig.tight_layout()

# fig.suptitle("RMSE")

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_rmse_useall.{fmt}", format=fmt, dpi=200)
