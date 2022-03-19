import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")


n_rep = 5

fontsize = 10
fontsize_small = 8
padding = 0.3

def type_to_problem(problem_type, **kwargs):
    if problem_type == "prangle_normal":
        return slad.PrangleNormalProblem(**kwargs)
    if problem_type == "prangle_gk":
        return slad.PrangleGKProblem(**kwargs)
    if problem_type == "prangle_lv":
        return slad.PrangleLVProblem(**kwargs)
    if problem_type == "fearnhead_gk":
        return slad.FearnheadGKProblem(**kwargs)
    if problem_type == "fearnhead_lv":
        return slad.FearnheadLVProblem(**kwargs)
    raise ValueError("Problem not recognized")


problem_types = [
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
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
    n_dist = len(slad.C2.distance_names)

    vals = np.full(shape=(n_dist, n_par, n_rep), fill_value=np.nan)

    for i_dist, distance_name in enumerate(slad.C2.distance_names):
        for i_rep in range(n_rep):
            h = pyabc.History(
                f"sqlite:///data_learn/{problem.get_id()}_{i_rep}/db_{distance_name}.db",
                create=False)

            df, w = h.get_distribution(t=h.max_t)
            vals[i_dist, :, i_rep] = np.array(
                [pyabc.weighted_mse(df[key], w, gt_par[key])
                 for key in gt_par])

            # print(vals[i_dist, i_mode, :, i_rep])

    means = np.mean(vals, axis=2)
    stds = np.std(vals, axis=2)

    #with open(pickle_file, "wb") as f:
    #    pickle.dump((means, stds, gt_par), f)

    return means, stds, gt_par


def plot_rmse(problem_type, log: bool, axes, ylabels: bool, width: float):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(slad.C2.distance_names)
    colors = list(slad.C2.distance_colors.values())
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist)-0.7)
            ax.set_yticklabels([
                slad.C2.distance_labels_short[dname] for dname in slad.C2.distance_names],
                fontdict={"fontsize": fontsize_small})
            #ax.xaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()
        ax.barh(
            ys - 0.7, means[:, i_par],
            #xerr=stds[:, 0, i_par],
            color=colors, alpha=0.9, height=0.7,
            error_kw={"ecolor": "grey"},
        )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            max_val = means[:, i_par].max()
            if log:
                pos_x = means[i_dist, i_par] * (1 + 1 / max_val)
            else:
                pos_x = means[i_dist, i_par] + (1  + 1 / max_val)
            ax.text(max_val * 0.99,
                    i_dist - 0.7,
                    f"{means[i_dist, i_par]:.3e}",
                    fontdict={"fontsize": fontsize_small},
                    verticalalignment="center",#"bottom" if i == 0 else "top",
                    horizontalalignment="right")

        #ax.set_xlabel("RMSE")
        ax.set_title(slad.C2.parameter_labels[problem_type][key], fontsize=fontsize)
        # ax.axhline(y=3.5, color="grey", linestyle="--")

        plt.setp(ax.get_xticklabels(), fontsize=fontsize_small)
        plt.setp(ax.get_xminorticklabels(), visible=False)

    # fig.suptitle(problem_labels[problem_type])
    axes[0].text(
        0, 1.1, slad.C2.problem_labels[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=12)

width_ratios = [3.5, 8, 5.5, 8, 5.5]
arr_cols = [1, 4, 3, 4, 3]
fig, axes = plt.subplots(nrows=1, ncols=sum(arr_cols), figsize=(16, len(slad.C2.distance_names) / 2), constrained_layout=True)

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_rmse(problem_type, log=True, axes=axes_for_problem, ylabels=i==0,
              width = width_ratios[i])

# fig.tight_layout()

# fig.suptitle("RMSE")

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_rmse.{fmt}", format=fmt, dpi=200)
