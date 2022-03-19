import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import scipy.stats as stats
import pickle
import argparse

import slad
from slad import C
import pyabc

pyabc.settings.set_figure_params("pyabc")

data_dir = "data_robust"
hist_suf = ""

fontsize_big = 12
fontsize_medium = 10
fontsize_small = 8
padding = 0.3

tumor_distance_names = [
    "Adaptive__Manhattan__mad_or_cmad",
    "Adaptive__Linear__Manhattan__mad_or_cmad__useall",
    "Adaptive__Linear__Manhattan__mad_or_cmad__Extend__useall",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__useall",
    "Info__Linear__Manhattan__mad_or_cmad__useall",
    "Info__Linear__Manhattan__mad_or_cmad__Extend__useall",
    "Info__MLP2__Manhattan__mad_or_cmad__Extend__useall",
]

def create_vals(problem_type):
    if problem_type == "tumor":
        problem = slad.TumorErrorProblem()
    else:
        raise ValueError()

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(tumor_distance_names)

    vals = np.full(shape=(n_dist, 2), fill_value=np.nan)

    for i_mode, kwargs in enumerate([{'n_obs_error': 0}, {}]):
        if problem_type == "tumor":
            if kwargs:
                problem = slad.TumorErrorProblem(noisy=True, frac_error=0)
            else:
                problem = slad.TumorErrorProblem(noisy=True, frac_error=0.1)
        else:
            raise ValueError()

        for i_dist, distance_name in enumerate(tumor_distance_names):
            if distance_name in tumor_distance_names:
                db_name = f"{data_dir}/{problem.get_id()}_0/db_{distance_name}.db"
                if not os.path.exists(db_name):
                    print(f"db {db_name} for error {kwargs} does not exist, continuing")
                    vals[i_dist, i_mode] = 0
                else:
                    h = pyabc.History("sqlite:///" + db_name, create=False)
                    _, w = h.get_distribution(t=h.max_t)
                    vals[i_dist, i_mode] = pyabc.effective_sample_size(w)

    return vals, gt_par


def plot_ess(
    problem_type,
    log: bool,
    axes,
    ylabels: bool,
):
    print(problem_type)
    means, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(tumor_distance_names)
    colors = [C.distance_colors_learn[dname] for dname in tumor_distance_names]

    for i_par in [0]:
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                C.distance_labels_short_learn[dname] for dname in tumor_distance_names],
                fontdict={"fontsize": fontsize_medium},
            )
            ax.yaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()
        ax.barh(
            ys - 0.2, means[:, 0],
            xerr=0,
            color=colors, alpha=0.3, height=0.4,
            error_kw={"ecolor": "grey", "alpha": 0.5},
        )
        ax.barh(
            ys + 0.2, means[:, 1],
            xerr=0,
            color=colors, alpha=0.8, height=0.4,
            error_kw={"ecolor": "grey", "alpha": 0.5},
        )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            for i in [0, 1]:
                max_val = means[:, :].max()
                if log:
                    pos_x = means[i_dist, i] * (1 + 1 / max_val)
                else:
                    pos_x = means[i_dist, i] + (1  + 1 / max_val)
                mean = means[i_dist, i]
                if mean == 0:
                    continue
                ax.text(max_val * 0.9,
                        i_dist - (-1)**i * 0.2,
                        f"{means[i_dist, i]:.1f}",
                        fontdict={"fontsize": fontsize_medium},
                        verticalalignment="center",
                        horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        #ax.set_title(slad.C.parameter_labels[problem_type][key], fontsize=fontsize_medium)
        #ax.axhline(y=3.5, color="grey", linestyle="dotted")

        plt.setp(ax.get_xticklabels(), fontsize=fontsize_small)
        plt.setp(ax.get_xminorticklabels(), visible=False)

    axes[0].text(
        0, 1.02, C.problem_labels_learn[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=fontsize_big,
    )

problem_types = ["tumor"]
arr_cols = [1]
fig, axes = plt.subplots(
    nrows=1, ncols=sum(arr_cols), figsize=(6, 5),
    #constrained_layout=True,
)
axes = [axes]

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_ess(
        problem_type=problem_type,
        log=False,
        axes=axes_for_problem,
        ylabels=i==0,
    )

fig.tight_layout(rect=(0.01, 0.1, 0.99, 0.99))
#plt.subplots_adjust(left=0.12, right=0.99, top=0.89, bottom=0.13)

# x axis label
axes[-1].set_xlabel("Effective sample size")
#fig.text(
#    0.5, 0.05, "Effective sample size",
#    horizontalalignment="center", verticalalignment="center",
#    fontsize=fontsize_medium,
#)

# legend
legend_elements = [
    Patch(facecolor="grey", alpha=0.3, label="Outlier-free"),
    Patch(facecolor="grey", alpha=0.8, label="Outlier-corrupted"),
]
axes[-1].legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, -0.12), ncol=2)
#plt.subplots_adjust(bottom=0.1)

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_tumor_ess{hist_suf}.{fmt}", format=fmt, dpi=200)
