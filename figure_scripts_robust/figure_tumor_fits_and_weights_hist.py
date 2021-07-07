import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

parser = argparse.ArgumentParser()
parser.add_argument("--frac_error", type=float, default=0)
args = parser.parse_args()
frac_error = args.frac_error
if frac_error == 0:
    frac_error = 0

distance_names = [
    "Calibrated__Euclidean__mad",
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
            arr_ax[i].plot(sum_stat[key], '-', color=color, alpha=5*1/5e2)

    def f_plot_mean(sum_stats, weights, arr_ax, **kwargs):
        aggregated = {}
        for key in sum_stats[0].keys():
            aggregated[key] = (np.array([sum_stat[key] for sum_stat in sum_stats]) \
                               * np.array(weights).reshape((-1,1))).sum(axis=0)
        for i, key in enumerate(aggregated.keys()):
            arr_ax[i].plot(aggregated[key], '-', color=color, alpha=1,
                           label=f"{slad.C.distance_labels_short[distance_name]}")

    if not show_distribution:
        f_plot = None
    if not show_mean:
        f_plot_mean = None

    pyabc.visualization.plot_data_callback(h, f_plot, f_plot_mean, ax=arr_ax)


fig = plt.figure(
    figsize=(14, 5),
    # constrained_layout=True
)

axes_all = fig.subplots(nrows=2, ncols=3)

problem_type = "tumor"
problem = slad.TumorErrorProblem(noisy=True, frac_error=frac_error)

data_dir = os.path.join("data_robust", f"{problem.get_id()}_0_p200")
db_dir = os.path.join("data_hist", f"{problem.get_id()}_0_p200")
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
    arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
    arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])

arr_ax[-1].legend()

#### weights

arr_ax = axes_all[1, :]

colors = [slad.C.distance_colors[dname] for dname in distance_names]

scale_log_files = []
for i_dist, distance_name in enumerate(distance_names):
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
        labels=[slad.C.distance_labels_short[dname] for dname in distance_names],
        ax=arr_ax[i_sumstat], colors=colors,
        keys_as_labels=False,
        keys=sumstat_keys,
    )

for i, key in enumerate(data.keys()):
    arr_ax[i].set_ylabel(slad.C.data_ylabels[problem_type][key])
    arr_ax[i].set_xlabel(slad.C.data_xlabels[problem_type][key])

for ax in arr_ax:
    ax.get_legend().remove()
arr_ax[-1].legend()

for ax in axes_all[0, :].flatten():
    ax.set_xlabel(None)

fig.tight_layout()

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_robust/figure_tumor_fits_and_weights_{frac_error}_hist.{fmt}", format=fmt, dpi=200)

