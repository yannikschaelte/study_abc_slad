import os
import matplotlib.pyplot as plt

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

distance_names = [
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__mad_or_cmad",
]

problem = slad.TumorErrorProblem(noisy=True, frac_error=0)
problem_error = slad.TumorErrorProblem(noisy=True, frac_error=0.1)
bounds = problem.get_prior_bounds()
refval = problem.get_gt_par()

pars = list(bounds.keys())

reduced = True
if reduced:
    pars = [
        "log_division_rate",
        "log_division_depth",
        "log_initial_spheroid_radius",
        "log_ecm_production_rate",
        "log_ecm_degradation_rate",
    ]

fig = plt.figure(
    figsize=(14, 5),
    constrained_layout=True,
)
arr_ax = fig.subplots(nrows=2, ncols=len(pars), gridspec_kw={"hspace": 0.15})

for distance_name in distance_names:
    h = pyabc.History(
        f"sqlite:///data_hist/{problem.get_id()}_0_p200/db_{distance_name}.db",
        create=False)
    df, w = h.get_distribution()
    for i_par, par in enumerate(pars):
        pyabc.visualization.plot_kde_1d(
            df, w, x=par, xmin=bounds[par][0], xmax=bounds[par][1],
            refval=refval, refval_color="grey", ax=arr_ax[0, i_par],
            color=slad.C.distance_colors[distance_name],
            label=slad.C.distance_labels_short[distance_name],
            xname=slad.C.parameter_labels["tumor"][par],
            numx=500,
        )

    h = pyabc.History(
        f"sqlite:///data_hist/{problem_error.get_id()}_0_p200/db_{distance_name}.db",
        create=False)
    df, w = h.get_distribution()
    for i_par, par in enumerate(pars):
        pyabc.visualization.plot_kde_1d(
            df, w, x=par, xmin=bounds[par][0], xmax=bounds[par][1],
            refval=refval, refval_color="grey", ax=arr_ax[1, i_par],
            color=slad.C.distance_colors[distance_name],
            label=slad.C.distance_labels_short[distance_name],
            xname=slad.C.parameter_labels["tumor"][par],
            numx=500,
        )

for ax in arr_ax[:, 1:].flatten():
    ax.set_ylabel(None)
for ax in arr_ax[0, :]:
    ax.set_xlabel(None)
arr_ax[0, -1].legend()

for i_err, label in enumerate(["No outliers", "20% of data points interchanged"]):
    arr_ax[i_err, 2].text(
        0.5, 1.1, label, transform=arr_ax[i_err, 2].transAxes,
        horizontalalignment="center", verticalalignment="center", fontsize=12)

# fig.tight_layout()

for fmt in ["pdf", "png"]:
    plt.savefig(f"figures_robust/figure_tumor_kdes_hist.{fmt}", format=fmt, dpi=200)
