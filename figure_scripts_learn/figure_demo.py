import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import slad
import pyabc
from slad import C

pyabc.settings.set_figure_params("pyabc")

i_rep = 0
data_dir_base = "data_learn_demo_useall"

distance_names = [
    "Adaptive",
    #"MS2",
    #"Adaptive__MS2",
    #"Adaptive__MS2__Extend",
    "Linear",
    "Adaptive__Linear",
    "Adaptive__Linear__Extend",
    "MLP2",
    "Adaptive__MLP2",
    "Adaptive__MLP2__Extend",
    #"Info__MS2",
    #"Info__MS2__Extend",
    "Info__Linear",
    "Info__Linear__Extend",
    "Info__MLP2",
    "Info__MLP2__Extend",
    #"Info2__MS2",
    #"Info2__MS2__Extend",
    #"Info2__Linear",
    #"Info2__Linear__Extend",
    #"Info2__MLP2",
    #"Info2__MLP2__Extend",
]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 5.5))

problem = slad.DemoProblem()
base_dir = os.path.join(data_dir_base, f"{problem.get_id()}_{i_rep}")

# true posterior
import scipy as sp
from functools import partial

def unnorm_1d_normal_pdf(p, y_obs, sigma, p_to_y = None):
    if p_to_y is None:
        p_to_y = lambda p: p
    y = p_to_y(p)
    pd = np.exp(- (y - y_obs)**2 / (2 * sigma**2))
    return pd

gt_par = problem.get_gt_par()
for i_par, (par, sigma) in enumerate(zip(gt_par, [1e-1, 1e2, 1e2, 1e-1])):
    p_to_y = lambda p: p
    if par == "p4":
        p_to_y = lambda p: p**2
    y_obs = p_to_y(gt_par[par])
    xmin, xmax = problem.get_prior_bounds()[par]
    pdf = partial(unnorm_1d_normal_pdf, y_obs=y_obs, sigma=sigma, p_to_y=p_to_y)
    norm = sp.integrate.quad(pdf, xmin, xmax)[0]
    xs = np.linspace(xmin, xmax, 300)
    for i_row in [0, 1]:
        axes[i_row, i_par].plot(
            xs, pdf(xs) / norm, linestyle="dashed", color="grey",
            label="ground truth")

# posteriors        
for dname in distance_names:
    i_row = 0
    if "Info" in dname or dname == "Adaptive":
        i_row = 1

    h = pyabc.History(
        "sqlite:///" + os.path.join(base_dir, f"db_{dname}.db"), create=False)

    for i_par, par in enumerate(problem.get_gt_par()):
        pyabc.visualization.plot_kde_1d_highlevel(
            h,
            x=par,
            xname=C.parameter_labels["demo"][par],
            xmin=problem.get_prior_bounds()[par][0],
            xmax=problem.get_prior_bounds()[par][1],
            #refval=problem.get_gt_par(),
            #refval_color="grey",
            color=C.distance_colors_learn_sep[dname],
            linestyle=":" if "MLP" in dname else "--" if "MS" in dname else "-",
            label=C.distance_labels_short_learn[dname],
            ax=axes[i_row, i_par],
            numx=300,
            kde=pyabc.GridSearchCV() if par == "p4" else None,
        )


for ax in axes[:, 1:].flatten():
    ax.set_ylabel(None)

#for i_row in [0, 1]:
#    axes[-1].axvline(-0.7, color="grey", linestyle="dotted")

titles = ["Regression-based summary statistics", "Regression-based sensitivity weights"]
for i_row in [0, 1]:
    axes[i_row, -1].legend(loc="center left", bbox_to_anchor=(1,0.5), title=titles[i_row])

#fig.text(
#   0.5, 0.99, "Regression-based summary statistics",
#   horizontalalignment="center", verticalalignment="top", fontsize=12)
#fig.text(
#   0.5, 0.51, "Regression-based distance information weights",
#   horizontalalignment="center", verticalalignment="top", fontsize=12)

fig.tight_layout()#rect=(0, 0, 1, 0.95))

# legend
#legend_elements = [
#    mpl.lines.Line2D(
#        [0], [0], color=distance_colors[dname], label=distance_labels[dname],
#    )
#    for dname in distance_names
#]
#legend_elements.append(mpl.lines.Line2D(
#    [0], [0], linestyle="dotted", color="grey", label="True parameters"))
#legend_elements.append(mpl.lines.Line2D(
#    [0], [0], linestyle="dashed", color="grey", label="True posterior"))
#axes[-1].legend(
#    handles=legend_elements,
#    loc="upper right",
#    bbox_to_anchor=(1, -0.2),
#    ncol=4,
#)

for fmt in ["png", "pdf"]:
    plt.savefig(f"figures_learn/figure_demo.{fmt}", format=fmt)
