import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import slad
import pyabc

pyabc.settings.set_figure_params("pyabc")

i_rep = 0
data_dir_base = "data_learn_demo"

distance_names = [
    "PNorm",
    "Adaptive",
    "Linear",
    "Adaptive__Linear",
    "Adaptive__Linear__Subset",
    "Info__Linear",
    "Info__Linear__Subset",
]

distance_labels = {
    "PNorm": "Unweighted",
    "Adaptive": "Adaptive",
    "Linear": "Unweighted Linear",
    "Adaptive__Linear": "Adaptive Linear",
    "Adaptive__Linear__Subset": "Adaptive Linear Subset",
    "Info__Linear": "Info Linear",
    "Info__Linear__Subset": "Info Linear Subset",
}

distance_colors = {key: f"C{i}" for i, key in enumerate(distance_names)}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

problem = slad.DemoProblem()
base_dir = os.path.join(data_dir_base, f"{problem.get_id()}_{i_rep}")

for dname in distance_names:
    h = pyabc.History(
        "sqlite:///" + os.path.join(base_dir, f"db_{dname}.db"), create=False)

    for i_par, par in enumerate(problem.get_gt_par()):
        pyabc.visualization.plot_kde_1d_highlevel(
            h,
            x=par,
            xname=par,
            xmin=problem.get_prior_bounds()[par][0],
            xmax=problem.get_prior_bounds()[par][1],
            refval=problem.get_gt_par(),
            refval_color="grey",
            color=distance_colors[dname],
            label=distance_labels[dname],
            ax=axes[i_par],
            numx=300,
            kde=pyabc.GridSearchCV(),
        )


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
for i_par, (par, sigma) in enumerate(zip(gt_par, [1e-1, 1e2, 1e-2])):
    p_to_y = lambda p: p
    if par == "p3":
        p_to_y = lambda p: p**2
    y_obs = p_to_y(gt_par[par])
    xmin, xmax = problem.get_prior_bounds()[par]
    pdf = partial(unnorm_1d_normal_pdf, y_obs=y_obs, sigma=sigma, p_to_y=p_to_y)
    norm = sp.integrate.quad(pdf, xmin, xmax)[0]
    xs = np.linspace(xmin, xmax, 300)
    axes[i_par].plot(xs, pdf(xs) / norm, linestyle="dashed", color="grey")


for ax in axes[1:]:
    ax.set_ylabel(None)

axes[-1].axvline(-0.7, color="grey", linestyle="dotted")

fig.tight_layout(rect=(0, 0.2, 1, 1))

# legend
legend_elements = [
    mpl.lines.Line2D(
        [0], [0], color=distance_colors[dname], label=distance_labels[dname],
    )
    for dname in distance_names
]
legend_elements.append(mpl.lines.Line2D(
    [0], [0], linestyle="dotted", color="grey", label="True parameters"))
legend_elements.append(mpl.lines.Line2D(
    [0], [0], linestyle="dashed", color="grey", label="True posterior"))
axes[-1].legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1, -0.2),
    ncol=4,
)

for fmt in ["png", "pdf"]:
    plt.savefig(f"figures_learn/figure_demo.{fmt}", format=fmt)
