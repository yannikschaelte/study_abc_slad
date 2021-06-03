import os
import matplotlib.pyplot as plt
import numpy as np

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Euclidean",
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Euclidean__cmad",
    "Adaptive__Euclidean__mad_or_cmad",
    "Manhattan",
    "Calibrated__Manhattan__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    "Info__Linear__Manhattan__mad_or_cmad",
]

distance_labels = {
    "Euclidean": "L2",
    "Calibrated__Euclidean__mad": "L2 + Cal. + MAD",
    "Adaptive__Euclidean__mad": "L2 + Adap. + MAD",
    "Adaptive__Euclidean__cmad": "L2 + Adap. + CMAD",
    "Adaptive__Euclidean__mad_or_cmad": "L2 + Adap. + (C)MAD",
    "Manhattan": "L1",
    "Calibrated__Manhattan__mad": "L1 + Cal. + MAD",
    "Adaptive__Manhattan__mad": "L1 + Adap. + MAD",
    "Adaptive__Manhattan__cmad": "L1 + Adap. + CMAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD",
    "Info__Linear__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD + Info",
}

problem_labels = {
    "gaussian": "Gaussian",
    "gk": "GK",
    "lv": "Lotka-Volterra",
}

data_dir = "data_robust"

for problem_type in [
    "gaussian",
    "gk",
    "lv",
]:
    print(problem_type)

    if problem_type == "gaussian":
        problem = slad.GaussianErrorProblem()
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem()
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem()

    gt_par = problem.get_gt_par()

    n_par = len(gt_par)
    n_dist = len(distance_names)

    vals = np.full(shape=(n_dist, 2, n_par), fill_value=np.nan)

    for i_mode, kwargs in enumerate([{'n_obs_error': 0}, {}]):

        if problem_type == "gaussian":
            problem = slad.GaussianErrorProblem(**kwargs)
        elif problem_type == "gk":
            problem = slad.PrangleGKErrorProblem(**kwargs)
        elif problem_type == "lv":        
            problem = slad.PrangleLVErrorProblem(**kwargs)

        for i_dist, distance_name in enumerate(distance_names):
            print(kwargs, distance_name)
            h = pyabc.History(
                f"sqlite:///data_robust/{problem.get_id()}/db_{distance_name}.db",
                create=False)

            df, w = h.get_distribution()
            vals[i_dist, i_mode] = np.array(
                [pyabc.weighted_rmse(df[key], w, gt_par[key]) for key in gt_par])

            print(vals[i_dist, i_mode])

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=n_par, figsize=(2+n_par*2, n_dist * 0.5))
    if n_par == 1:
        axes = [axes]
    for i_par, key in enumerate(gt_par.keys()):
        ax = axes[i_par]
        ys = np.arange(n_dist)
        if i_par == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([distance_labels[name] for name in distance_names])
        else:
            ax.set_yticks([])
        ax.invert_yaxis()
        ax.barh(ys - 0.2, vals[:, 0, i_par], color='C0', height=0.4)
        ax.barh(ys + 0.2, vals[:, 1, i_par], color='C1', height=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("RMSE")
        ax.set_title(key)

    fig.suptitle(problem_labels[problem_type])
    fig.tight_layout()
    plt.savefig(f"plot_robust/rmse_{problem_type}.png")
