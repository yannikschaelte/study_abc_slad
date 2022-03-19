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
    if problem_type == "demo":
        return slad.DemoProblem(**kwargs)
    if problem_type == "cr":
        return slad.CRProblem(**kwargs)
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
    if problem_type == "harrison_toy":
        return slad.HarrisonToyProblem(**kwargs)
    raise ValueError(f"Problem not recognized: {problem_type}")


problem_types = [
    "demo",
    "cr",
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
    "harrison_toy",
]

problem_labels = {key: ix for ix, key in enumerate(problem_types)}

distance_names = [
    "Adaptive",
    "Linear__Subset",
    "Adaptive__Linear__Subset",
    "Info__Linear__Subset",
]

distance_labels_short = {key: key for key in distance_names}
distance_colors = {key: f"C{i}" for i, key in enumerate(distance_names)}

fracs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7]
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
                if distance_name in ["PNorm", "Adaptive"]:
                    frac = 0.0
                h = pyabc.History(
                    f"sqlite:///data_learn_pilot/{problem.get_id()}_{i_rep}/db_frac_{frac}_{distance_name}.db",
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


def plot_rmse(problem_type, log: bool, axes, ylabels: bool):
    print(problem_type)
    means, stds, gt_par = create_vals(problem_type)
    n_par = len(gt_par)

    n_dist = len(distance_names)
    colors = [distance_colors[dname] for dname in distance_names]
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if ylabels and i_par == 0:
            yticks = []
            yticklabels = []
            for i_dist, dist in enumerate(distance_names):
                for i_frac, frac in enumerate(fracs):
                    if i_frac == 0:
                        label = distance_labels_short[dist] + " - "
                    else:
                        label = ""
                    label += f"{100*frac:.0f}%"
                    yticklabels.append(label)
                    yticks.append(i_dist + 0.9 * (i_frac / n_frac))
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontdict={"fontsize": fontsize_small})
            #ax.set_yticks(np.arange(n_dist))
            #ax.set_yticklabels([
            #    distance_labels_short[dname] for dname in distance_names],
            #    fontdict={"fontsize": fontsize_small})
            #ax.xaxis.set_ticks_position("none")
        else:
            ax.set_yticks([])

        ax.invert_yaxis()

        for i_frac, frac in enumerate(fracs):
            ax.barh(
                ys + 0.9 * (i_frac / n_frac), means[:, i_frac, i_par],
                #xerr=stds[:, 0, i_par],
                color=colors, alpha=(i_frac+1) / n_frac, height=0.9 / n_frac,
                error_kw={"ecolor": "grey"},
            )
        if log:
            ax.set_xscale("log")

        # add value
        for i_dist in range(n_dist):
            max_val = means[:, :, i_par].max()
            for i_frac, frac in enumerate(fracs):
                ax.text(
                    max_val * 0.99,
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
        0, 1.1, problem_labels[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=12)

arr_cols = [3, 2, 1, 4, 3, 4, 3, 1]
fig, axes = plt.subplots(nrows=1, ncols=sum(arr_cols), figsize=(16, 5), constrained_layout=True)

for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[sum(arr_cols[:i]):sum(arr_cols[:i+1])]
    plot_rmse(problem_type, log=True, axes=axes_for_problem, ylabels=i==0)

# fig.tight_layout()

# fig.suptitle("RMSE")

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_rmse_pilot.{fmt}", format=fmt, dpi=200)
