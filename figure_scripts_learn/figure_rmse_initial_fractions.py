import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")


n_rep = 1

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
    #"prangle_gk",
    #"prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
]

distance_names = [
    #"Adaptive",
    "Linear__Initial",
    #"Adaptive__Linear__Initial",
    #"Adaptive__Linear",
    #"GP__Initial",
    #"Adaptive__GP__Initial",
    "MLP__Initial",
    #"Adaptive__MLP__Initial",
    "Info__Linear__Initial",
    #"Info__Linear",
]
distance_labels_short = {key: key for key in distance_names}
distance_colors = {key: f"C{i}" for i, key in enumerate(distance_names)}

fracs = [0.1, 0.2, 0.5, 0.7]
n_frac = len(fracs)

def create_vals(problem_type):
    pickle_file = f"figures_learn/data_frac_{problem_type}.pickle"
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            means, stds, gt_par = pickle.load(f)
            return means, stds, gt_par

    problem = type_to_problem(problem_type)

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(distance_names)

    vals = np.full(shape=(n_dist, n_frac, n_par, n_rep), fill_value=np.nan)

    for i_dist, distance_name in enumerate(distance_names):
        for i_frac, frac in enumerate(fracs):
            for i_rep in range(n_rep):
                h = pyabc.History(
                    f"sqlite:///data_learn/{problem.get_id()}_{i_rep}/db_frac_{frac}_{distance_name}.db",
                    create=False)

                df, w = h.get_distribution(t=h.max_t)
                vals[i_dist, i_frac, :, i_rep] = np.array(
                    [pyabc.weighted_rmse(df[key], w, gt_par[key])
                     for key in gt_par])

            # print(vals[i_dist, i_mode, :, i_rep])

    means = np.mean(vals, axis=3)
    stds = np.std(vals, axis=3)

    #with open(pickle_file, "wb") as f:
    #    pickle.dump((means, stds, gt_par), f)

    return means, stds, gt_par


def plot_rmse(problem_type, log: bool, axes, ylabels: bool, width: float):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(distance_names)
    colors = [distance_colors[dname] for dname in distance_names]
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            ax.set_yticks(np.arange(n_dist))
            ax.set_yticklabels([
                distance_labels_short[dname] for dname in distance_names],
                fontdict={"fontsize": fontsize_small})
            #ax.xaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()

        for i_frac, frac in enumerate(fracs):
            ax.barh(
                ys + 0.9 * (i_frac / n_frac), means[:, i_frac, i_par],
                #xerr=stds[:, 0, i_par],
                color=colors, alpha=0.9, height=0.9 / n_frac,
                error_kw={"ecolor": "grey"},
            )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            max_val = means[:, :, i_par].max()
            for i_frac, frac in enumerate(fracs):
                ax.text(max_val * 0.99,
                    i_dist + 0.9 * (i_frac / n_frac),
                    f"{means[i_dist, i_frac, i_par]:.5f}",
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

width_ratios = [3.3, 8, 5.5]
arr_cols = [1, 4, 3]
fig, axes = plt.subplots(nrows=1, ncols=sum(arr_cols), figsize=(14, 4), constrained_layout=True)

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_rmse(problem_type, log=True, axes=axes_for_problem, ylabels=i==0,
              width = width_ratios[i])

# fig.tight_layout()

# fig.suptitle("RMSE")

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_rmse_initial_fractions.{fmt}", format=fmt, dpi=200)
