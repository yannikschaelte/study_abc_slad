import os
import numpy as np
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

alpha = 1 / 1e2

distance_names = [
    #"Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


def plot_sumstats(
    h, data, problem_type, distance_name, arr_ax,
    show_distribution=True, show_mean=True,
):
    color = slad.C.distance_colors[distance_name]
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], '-', color=color, alpha=alpha)
    def f_plot_lv(sum_stat, weight, arr_ax, **kwargs):
        arr_ax[0].plot(sum_stat["y"][:, 0], '-', color=color, alpha=alpha)
        arr_ax[1].plot(sum_stat["y"][:, 1], '-', color=color, alpha=alpha)

    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color=color, alpha=1,
                           label=f"{slad.C.distance_labels_short[distance_name]}")
    def f_plot_mean_lv(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        aggregated["Prey"] = (np.array([sum_stat["y"][:, 0] for sum_stat in sum_stats]) \
                              * np.array(weights).reshape((-1,1))).sum(axis=0)
        aggregated["Predator"] = (np.array([sum_stat["y"][:, 1] for sum_stat in sum_stats]) \
                                  * np.array(weights).reshape((-1,1))).sum(axis=0)
        arr_ax[0].plot(
            aggregated["Prey"], '-', color=color, alpha=1,
            label=f"{slad.C.distance_labels_short[distance_name]}")
        arr_ax[1].plot(
            aggregated["Predator"], '-', color=color, alpha=1,
            label=f"{slad.C.distance_labels_short[distance_name]}")

    if problem_type == "lv":
        f_plot = f_plot_lv
        f_plot_mean = f_plot_mean_lv

    if not show_distribution:
        f_plot = None
    if not show_mean:
        f_plot_mean = None

    pyabc.visualization.plot_data_callback(h, f_plot, f_plot_mean, ax=arr_ax)

fig = plt.figure(
    figsize=(14, 5),
    #constrained_layout=True,
)
axes_all = fig.subplots(nrows=2, ncols=6, gridspec_kw={"width_ratios": [3, 3, 3, 3, 2, 2]})

configs = [
    ("uninf", 0),
    ("gaussian", 19),
    ("cr-zero", 2),
    ("gk", 16),
    ("lv", 1),
]

axes_fits = [[axes_all[0, 0]], [axes_all[0, 1]], [axes_all[0, 2]], [axes_all[0, 3]], axes_all[0, 4:]]

for i_subfig, (arr_ax, (problem_type, i_rep)) in enumerate(zip(axes_fits, configs)):

    kwargs = {}
    if problem_type == "uninf":
        problem = slad.UninfErrorProblem(**kwargs)
    elif problem_type == "gaussian":
        problem = slad.GaussianErrorProblem(**kwargs)
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem(**kwargs)
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem(**kwargs)
    elif problem_type == "cr-zero":
        problem = slad.CRErrorZeroProblem(**kwargs)
    elif problem_type == "cr-swap":
        problem = slad.CRErrorSwapProblem(**kwargs)

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")
    db_dir = os.path.join(dir, "..", "data_hist", f"{problem.get_id()}_{i_rep}")
    data = load_data(problem, data_dir)

    for distance_name in distance_names:
        h = pyabc.History(
            "sqlite:///" + os.path.join(db_dir, f"db_{distance_name}.db"),
            create=False)
        plot_sumstats(h, data, problem_type, distance_name, arr_ax, show_mean=False)

    for distance_name in distance_names:
        h = pyabc.History(
            "sqlite:///" + os.path.join(db_dir, f"db_{distance_name}.db"),
            create=False)
        plot_sumstats(h, data, problem_type, distance_name, arr_ax, show_distribution=False)

    # data and labels
    if problem_type != "lv":
        for i, key in enumerate(data.keys()):
            arr_ax[i].plot(data[key], 'x', color="black", label='Observed data')
            arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
            #arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])
    else:
        arr_ax[0].plot(data["y"][:, 0], 'x', color="black", label='Observed data')
        arr_ax[0].set_ylabel("# Prey")
        #arr_ax[0].set_xlabel("Time [au]")
        arr_ax[1].plot(data["y"][:, 1], 'x', color="black", label='Observed data')
        arr_ax[1].set_ylabel("# Predator")
        #arr_ax[1].set_xlabel("Time [au]")

    if i_subfig == 1:
        arr_ax[0].legend()

    arr_ax[0].text(
        0, 1.1, slad.C.problem_labels[problem_type],
        horizontalalignment="left", verticalalignment="bottom",
        transform=arr_ax[0].transAxes, fontsize=12)
    # subfig.suptitle(slad.C.problem_labels[problem_type])


axes_weights = [[axes_all[1, 0]], [axes_all[1, 1]], [axes_all[1, 2]], [axes_all[1, 3]], axes_all[1, 4:]]

for i_subfig, (arr_ax, (problem_type, i_rep)) in enumerate(zip(axes_weights, configs)):
    colors = [slad.C.distance_colors[dname] for dname in distance_names]

    kwargs = {}
    if problem_type == "uninf":
        problem = slad.UninfErrorProblem(**kwargs)
    elif problem_type == "gaussian":
        problem = slad.GaussianErrorProblem(**kwargs)
    elif problem_type == "gk":
        problem = slad.PrangleGKErrorProblem(**kwargs)
    elif problem_type == "lv":
        problem = slad.PrangleLVErrorProblem(**kwargs)
    elif problem_type == "cr-zero":
        problem = slad.CRErrorZeroProblem(**kwargs)
    elif problem_type == "cr-swap":
        problem = slad.CRErrorSwapProblem(**kwargs)

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")
    db_dir = os.path.join(dir, "..", "data_hist", f"{problem.get_id()}_{i_rep}")
    data = load_data(problem, data_dir)

    scale_log_files = []
    for i_dist, distance_name in enumerate(distance_names):
        scale_log_file = os.path.join(data_dir, f"log_scale_{distance_name}.json")
        scale_log_files.append(scale_log_file)

    if problem_type != "lv":
        pyabc.visualization.plot_distance_weights(
            scale_log_files,
            labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
            ax=arr_ax[0], colors=colors,
            keys_as_labels=False)
    else:
        keys = [f"y:{i}" for i in range(0, 16)]
        pyabc.visualization.plot_distance_weights(
            scale_log_files,
            labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
            ax=arr_ax[0], colors=colors, keys=keys, keys_as_labels=False)
        keys = [f"y:{i}" for i in range(16, 32)]
        pyabc.visualization.plot_distance_weights(
            scale_log_files,
            labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
            ax=arr_ax[1], colors=colors, keys=keys, keys_as_labels=False)

    # labels
    if problem_type != "lv":
        for i, key in enumerate(data.keys()):
            arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
            arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])
    else:
        arr_ax[0].set_ylabel("# Prey")
        arr_ax[0].set_xlabel("Time [au]")
        arr_ax[1].set_ylabel("# Predator")
        arr_ax[1].set_xlabel("Time [au]")

    for ax in arr_ax:
        ax.get_legend().remove()
        ax.set_ylabel(None)
    if i_subfig == 0:
        arr_ax[0].set_ylabel("Weight")
    if i_subfig == 1:
        arr_ax[0].legend()

    # subfig.suptitle(slad.C.problem_labels[problem_type])

fig.tight_layout()

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_robust/figure_fits_and_weights_hist.{fmt}", format=fmt, dpi=200)

