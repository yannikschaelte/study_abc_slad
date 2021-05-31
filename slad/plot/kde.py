import matplotlib.pyplot as plt
import pyabc


def plot_1d_kdes(problem, labels, iters):
    bounds = problem.get_prior_bounds()
    pars = list(bounds.keys())
    refval = problem.get_gt_par()

    n_label = len(labels)
    n_par = len(pars)

    fig, axes = plt.subplots(
        nrows=n_label, ncols=n_par, figsize=(3 * n_par, 3 * n_label)
    )
    axes = axes.reshape(n_label, n_par)

    for i_label, label in enumerate(labels):
        for iter in iters:
            h = pyabc.History(
                f"sqlite:///data/{problem.get_id()}/db_{label}_{iter}.db", create=False
            )
            df, w = h.get_distribution()
            for i_par, par in enumerate(pars):
                pyabc.visualization.plot_kde_1d(
                    df,
                    w,
                    x=par,
                    xmin=bounds[par][0],
                    xmax=bounds[par][1],
                    refval=refval,
                    refval_color="grey",
                    ax=axes[i_label, i_par],
                )
        for i_par in range(n_par):
            if i_par == 0:
                axes[i_label, i_par].set_ylabel(label)
            else:
                axes[i_label, i_par].set_ylabel(None)
            if i_label < n_label - 1:
                axes[i_label, i_par].set_xlabel(None)

    fig.tight_layout()

    return fig, axes
