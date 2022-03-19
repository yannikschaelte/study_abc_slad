import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

import slad
from slad import C
import pyabc

parser = argparse.ArgumentParser()
parser.add_argument("--frac_error", type=float)
parser.add_argument("--hist", type=int)
args = parser.parse_args()
use_hist = args.hist
if use_hist == 1:
    db_dir_base = "data_hist"
    hist_suf = "_hist"
elif use_hist == 0:
    db_dir_base = "data_robust"
    hist_suf = ""

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    #"Calibrated__Euclidean__mad",
    #"Adaptive__Euclidean__mad",
    #"Adaptive__Manhattan__mad",
    #"Adaptive__Manhattan__mad_or_cmad",
    #"Adaptive__MS__Manhattan__mad_or_cmad",
    #"Adaptive__MS2__Manhattan__mad_or_cmad__Extend",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
    #"Info__MS__Manhattan__mad_or_cmad",
    #"Info__MS2__Manhattan__mad_or_cmad__Extend",
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

problem = slad.TumorErrorProblem(noisy=True, frac_error=0)
problem_error = slad.TumorErrorProblem(noisy=True, frac_error=0.1)
bounds = problem.get_prior_bounds()
refval = problem.get_gt_par()

#bounds["log_division_rate"] = (-2, -1)
bounds["log_ecm_production_rate"] = (-2.8, -2)

pars = list(bounds.keys())

reduced = False
if reduced:
    pars = [
        "log_division_rate",
        "log_division_depth",
        "log_initial_spheroid_radius",
        "log_ecm_production_rate",
        "log_ecm_degradation_rate",
    ]

fig = plt.figure(
    figsize=(12, 5),
    #constrained_layout=True,
)
arr_ax = fig.subplots(
    nrows=2,
    ncols=len(pars),
    #gridspec_kw={"hspace": 0.15},
)

for distance_name in distance_names:
    db_name = f"{db_dir_base}/{problem.get_id()}_0/db_{distance_name}.db"
    h = pyabc.History(
        "sqlite:///" + db_name,
        create=False)
    df, w = h.get_distribution()
    for i_par, par in enumerate(pars):
        pyabc.visualization.plot_kde_1d(
            df, w, x=par, xmin=bounds[par][0], xmax=bounds[par][1],
            refval=refval, refval_color="grey", ax=arr_ax[0, i_par],
            color=C.distance_colors_learn[distance_name],
            label=C.distance_labels_short_learn[distance_name],
            xname=C.parameter_labels["tumor"][par],
            linestyle=":" if "MLP2" in distance_name else "-",
            numx=500,
        )

    db_name = f"{db_dir_base}/{problem_error.get_id()}_0/db_{distance_name}.db"
    if not os.path.exists(db_name):
        print(f"db {db_name} for error does not exist, continuing")
        continue
    h = pyabc.History(
        "sqlite:///" + db_name,
        create=False)
    df, w = h.get_distribution()
    for i_par, par in enumerate(pars):
        pyabc.visualization.plot_kde_1d(
            df, w, x=par, xmin=bounds[par][0], xmax=bounds[par][1],
            refval=refval, refval_color="grey", ax=arr_ax[1, i_par],
            color=C.distance_colors_learn[distance_name],
            label=C.distance_labels_short_learn[distance_name],
            xname=C.parameter_labels["tumor"][par],
            linestyle=":" if "MLP2" in distance_name else "-",
            numx=500,
        )

for ax in arr_ax[:, 1:].flatten():
    ax.set_ylabel(None)
arr_ax[0, 0].set_ylabel("ABC posterior")
arr_ax[1, 0].set_ylabel("ABC posterior")
for ax in arr_ax.flatten():
    ax.set_xlabel(None)
for ax, par in zip(arr_ax[0, :], pars):
    ax.set_title(C.parameter_labels["tumor"][par], fontsize=10)

fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.98), h_pad=2) # left, bottom, right, top

fig.text(
   0.5, 0.99, "Tumor model, Outlier-free data",
   horizontalalignment="center", verticalalignment="top", fontsize=12)
fig.text(
   0.5, 0.53, "Tumor model, Outlier-corrupted data",
   horizontalalignment="center", verticalalignment="top", fontsize=12)

# legend
legend_elements = [
    mpl.lines.Line2D(
        [0], [0],
        color=C.distance_colors_learn[dname],
        label=C.distance_labels_short_learn[dname],
        linestyle=":" if "MLP2" in dname else "-",
    )
    for dname in reversed(distance_names)
]
arr_ax.flatten()[-1].legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1, -0.14),
    ncol=4,
)

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_learn/figure_tumor_kdes{hist_suf}_all.{fmt}", format=fmt, dpi=200)
