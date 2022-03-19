import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

import slad
from slad import C
import pyabc

pyabc.settings.set_figure_params("pyabc")

parser = argparse.ArgumentParser()
parser.add_argument("--frac_error", type=float)
parser.add_argument("--hist", type=int)
args = parser.parse_args()
frac_error = args.frac_error
use_hist = args.hist
if frac_error == 0:
    frac_error = 0
if use_hist == 1:
    db_dir_base = "data_hist"
    hist_suf = "_hist"
elif use_hist == 0:
    db_dir_base = "data_robust"
    hist_suf = ""

distance_names = [
    #"Calibrated__Euclidean__mad",
    #"Adaptive__Euclidean__mad",
    #"Adaptive__Manhattan__mad",
    #"Adaptive__Manhattan__mad_or_cmad",
    #"Adaptive__Linear__Manhattan__mad_or_cmad",
    #"Adaptive__MS__Manhattan__mad_or_cmad",
    #"Adaptive__MS2__Manhattan__mad_or_cmad__Extend",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
    #"Info__MS__Manhattan__mad_or_cmad",
    #
    #"Adaptive__MS2__Manhattan__mad_or_cmad__Extend",
    "Info__MLP2__Manhattan__mad_or_cmad__Extend__useall",
    "Info__Linear__Manhattan__mad_or_cmad__Extend__useall",
    "Info__Linear__Manhattan__mad_or_cmad__useall",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__useall",
    "Adaptive__Linear__Manhattan__mad_or_cmad__Extend__useall",
    "Adaptive__Linear__Manhattan__mad_or_cmad__useall",
    "Adaptive__Manhattan__mad_or_cmad",
]
weighted_distance_names = [distance_names[i] for i in [0, 1, 2, 6]]
info_distance_names=distance_names[0:3]

if frac_error == 0.1:
    distance_names = [distance_names[i] for i in [1, 6]]
    weighted_distance_names = distance_names
    info_distance_names = [distance_names[0]]


# constants
alpha = 0.05
n_sample = 20
fontsize_large = 12
fontsize_plus = 11
fontsize_medium = 10

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
    color = slad.C.distance_colors_learn[distance_name]
    linestyle = ":" if "MLP2" in distance_name else "-"
    def f_plot(sum_stat, weight, arr_ax, **kwargs):
        for i, key in enumerate(sum_stat.keys()):
            arr_ax[i].plot(sum_stat[key], color=color, linestyle=linestyle, alpha=alpha)

    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], color=color, linestyle=linestyle, alpha=1,
                           label=f"{C.distance_labels_short_learn[distance_name]}")

    if not show_distribution:
        f_plot = None
    if not show_mean:
        f_plot_mean = None

    pyabc.visualization.plot_data_callback(
        h, f_plot, f_plot_mean, ax=arr_ax, n_sample=n_sample)


fig = plt.figure(
    figsize=(12, 6.5),
    # constrained_layout=True
)

axes_all = fig.subplots(nrows=3, ncols=3)

problem_type = "tumor"
problem = slad.TumorErrorProblem(noisy=True, frac_error=frac_error)

data_dir = os.path.join("data_robust", f"{problem.get_id()}_0")
db_dir = os.path.join(db_dir_base, f"{problem.get_id()}_0")
data = load_data(problem, data_dir)

arr_ax = axes_all[0, :]

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
for i, key in enumerate(data.keys()):
    arr_ax[i].plot(data[key], 'x', color="black", label='Observed data')
    arr_ax[i].set_title(slad.C.data_ylabels[problem_type][key], fontsize=fontsize_medium)
    arr_ax[i].set_xlabel(None)
    arr_ax[i].set_ylabel(None)

# arr_ax[-1].legend()

#### weights

arr_ax = axes_all[1, :]

scale_log_files = []
for i_dist, distance_name in enumerate(weighted_distance_names):
    scale_log_file = os.path.join(data_dir, f"log_scale_{distance_name}.json")
    scale_log_files.append(scale_log_file)

sumstat_keys_arr = [
    [f"growth_curve:{i}" for i in range(20)],
    [f"extra_cellular_matrix_profile:{i}" for i in range(65)],
    [f"proliferation_profile:{i}" for i in range(65)],
]

for i_sumstat, sumstat_keys in enumerate(sumstat_keys_arr):
    pyabc.visualization.plot_distance_weights(
        scale_log_files,
        labels=[C.distance_labels_short_learn[dname] for dname in weighted_distance_names],
        ax=arr_ax[i_sumstat],
        colors=[C.distance_colors_learn[dname] for dname in weighted_distance_names],
        keys_as_labels=False,
        keys=sumstat_keys,
        linestyles=[":" if "MLP2" in dname else "-" for dname in weighted_distance_names],
    )

for i, key in enumerate(data.keys()):
    arr_ax[i].set_ylabel(None)
    arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])

for ax in arr_ax:
    ax.get_legend().remove()
    ax.set_ylim(bottom=0)

# info weights

arr_ax = axes_all[2, :]

info_log_files = [
    os.path.join(data_dir, f"log_info_{dname}.json") for dname in info_distance_names
]
for i_sumstat, sumstat_keys in enumerate(sumstat_keys_arr):
    pyabc.visualization.plot_distance_weights(
        info_log_files,
        labels=[C.distance_labels_short_learn[dname] for dname in info_distance_names],
        ax=arr_ax[i_sumstat],
        colors=[C.distance_colors_learn[dname] for dname in info_distance_names],
        keys_as_labels=False,
        keys=sumstat_keys,
        linestyles=[":" if "MLP2" in dname else "-" for dname in info_distance_names],
    )

for i, key in enumerate(data.keys()):
    arr_ax[i].set_ylabel(None)
    arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])

for ax in arr_ax:
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_ylim(bottom=0)

# stuff

for ax in axes_all[0, :].flatten():
    ax.set_xlabel(None)
for ax in axes_all[1, :].flatten():
    ax.set_xlabel(None)
    
# same ticks
for ax in axes_all.flatten():
    ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())

fig.tight_layout(rect=(0.02, 0.08, 1, 0.95)) # left, bottom, right, top

if frac_error > 0:
    title = "Tumor model, Outlier-corrupted data"
else:
    title = "Tumor model, Outlier-free data"
axes_all[0, 1].text(
    0.5, 1.3, title,
    verticalalignment="top", horizontalalignment="center",
    transform=axes_all[0, 1].transAxes,
    fontsize=fontsize_large,
)
axes_all[0, 0].set_ylabel("Data and simulations")
axes_all[1, 0].set_ylabel("Scale weights")
axes_all[2, 0].set_ylabel("Sensitivity weights")

# legend
legend_elements = [
    mpl.lines.Line2D(
        [0], [0],
        color=slad.C.distance_colors_learn[dname],
        label=slad.C.distance_labels_short_learn[dname],
        linestyle=":" if "MLP2" in dname else "-",
    )
    for dname in reversed(distance_names)
]
#legend_elements.append(mpl.lines.Line2D(
#    [0], [0], color="grey", linestyle="-", label="Scale weights"))
#legend_elements.append(mpl.lines.Line2D(
#    [0], [0], color="grey", linestyle=":", label="Sensitivity weights"))
legend_elements.append(mpl.lines.Line2D(
    [0], [0], marker="x", color="black", linestyle="None", label="Observed data"))
axes_all.flatten()[-1].legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1, -0.35),
    ncol=4,
)

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_tumor_fits_and_weights_{frac_error}{hist_suf}.{fmt}", format=fmt, dpi=200)

