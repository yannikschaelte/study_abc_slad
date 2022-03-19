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
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
    "harrison_toy",
]

problem_labels = {key: ix for ix, key in enumerate(problem_types)}

distance_names = [
    "Linear__Subset",
    "Info__Linear__Subset",
]

distance_labels_short = {key: key for key in distance_names}
distance_colors = {key: f"C{i}" for i, key in enumerate(distance_names)}

fracs = [0.0, 0.2, 0.3, 0.4, 0.7]
n_frac = len(fracs)


def plot_kdes(problem_type, axes):
    print(problem_type)
    problem = type_to_problem(problem_type)
    gt_par = problem.get_gt_par()
    n_par = len(gt_par)

    n_dist = len(distance_names)
    colors = [distance_colors[dname] for dname in distance_names]
    i_rep = 0

    for i_dist, dist in enumerate(distance_names):
        for i_frac, frac in enumerate(fracs):
            h = pyabc.History(
                f"sqlite:///data_learn_pilot/{problem.get_id()}_{i_rep}/db_frac_{frac}_{dist}.db",
                create=False)
            for i_par, par in enumerate(gt_par.keys()):
                ax = axes[i_par]
                pyabc.visualization.plot_kde_1d_highlevel(
                    h, x=par,
                    xmin=problem.get_viz_bounds()[par][0],
                    xmax=problem.get_viz_bounds()[par][1],
                    refval=gt_par, refval_color="grey",
                    ax=ax,
                    numx=200,
                    color=distance_colors[dist],
                    label=f"{dist} - {100*frac:.0f}%",
                    alpha=(i_frac+1)/n_frac,
                )
            for i_par in range(n_par, 4):
                axes[i_par].axis("off")

    # fig.suptitle(problem_labels[problem_type])
    axes[0].text(
        0, 1.1, problem_labels[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=axes[0].transAxes, fontsize=12)

arr_cols = [3, 1, 4, 3, 4, 3, 1]
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(12, 4 * 7), constrained_layout=True)
for i, (problem_type, cols) in enumerate(zip(problem_types, arr_cols)):
    axes_for_problem = axes[i]
    plot_kdes(problem_type, axes=axes_for_problem)

#axes[1,0].legend(loc="center", bbox_to_anchor=(1, 0.5))
# fig.tight_layout()

# fig.suptitle("RMSE")

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_kdes_pilot.{fmt}", format=fmt, dpi=200)
