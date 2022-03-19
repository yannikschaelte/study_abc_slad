import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#from plotly.offline import init_notebook_mode
#init_notebook_mode()
#import plotly.io as pio
#print(pio.kaleido.scope.mathjax)
#pio.kaleido.scope.mathjax = "cdn"

import pyabc
from pyabc.predictor import *
from pyabc.sumstat import *
from pyabc.util import read_sample, ParTrafo
from pyabc.distance.util import fd_nabla1_multi_delta
import slad

pyabc.settings.set_figure_params("pyabc")

problem = slad.DemoProblem()
i_rep = 0
data_dir_base = "data_learn_demo_useall"
base_dir = os.path.join(data_dir_base, f"{problem.get_id()}_{i_rep}")

dname = "Info__Linear__Extend"

# extract samples

h = pyabc.History(
    "sqlite:///" + os.path.join(base_dir, f"db_{dname}.db"),
    create=False,
)

# exract latest fitting time point
info_log_file = os.path.join(base_dir, f"log_info_{dname}.json")
info_dict = pyabc.storage.load_dict_from_json(info_log_file)
t = max(info_dict.keys())

# get data matrix

# sample from history
sample = pyabc.Sample.from_population(h.get_population(t=t-1))


# read sample
par_keys = list(h.get_distribution()[0].columns)
pre = IdentitySumstat()

pre.initialize(
    t=0, get_sample=lambda: sample,
    x_0=h.observed_sum_stat(), total_sims=0)

par_trafo = ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4])
par_trafo.initialize(par_keys)

x, y, w = read_sample(
    sample=sample, sumstat=pre, all_particles=False, par_trafo=par_trafo,
)
x, y, w = [np.load(f"{base_dir}/log_info_sample_{dname}_{t}_{var}.npy")
           for var in ["sumstats", "parameters", "weights"]]

# get observed data
s_0 = pre(h.observed_sum_stat())

# subset
#subsetter = GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10})
#x, y, w = subsetter.select(x, y, w)

# apply normalization
offset_x = np.nanmedian(x, axis=0)
scale_x = np.nanmedian(np.abs(x - offset_x), axis=0)
x = (x - offset_x) / scale_x
x0 = (s_0 - offset_x) / scale_x
y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

for s_predictor, predictor_label in [
    ("Lin", "linear regression"),
    ("MLP2", "a neural network"),
    #("LASSO", "LASSO"),
    #("GP", "a Gaussian process"),
]:
    np.random.seed(2)

    # fit predictor
    if s_predictor == "Lin":
        predictor = LinearPredictor()
    elif s_predictor == "MLP2":
        predictor = MLPPredictor(
            alpha=1,
            hidden_layer_sizes=HiddenLayerHandle(method=["mean"]),
            solver="adam",
            max_iter=20000,
            n_iter_no_change=10,
            verbose=True,
        )
    elif s_predictor == "LASSO":
        predictor = LassoPredictor()
    elif s_predictor == "GP":
        predictor = GPPredictor(
            kernel=GPKernelHandle(ard=False),
        )

    predictor.fit(x=x, y=y, w=w)

    def fun(_x):
        """Predictor function."""
        return predictor.predict(_x.reshape(1, -1)).flatten()

    # calculate sensitivities
    sensis = fd_nabla1_multi_delta(
        x=x0, fun=fun, test_deltas=None)
    n_x = x.shape[1]
    n_y = y.shape[1]
    if sensis.shape != (n_x, n_y):
        raise AssertionError("Sensitivity shape did not match.")

    # only interested in absolute values
    sensis = np.abs(sensis)

    # total sensitivities per parameter
    sensi_per_y = np.sum(sensis, axis=0)

    # normalize per parameter to 1
    y_has_sensi = ~np.isclose(sensi_per_y, 0.)
    sensis[:, ~y_has_sensi] = 0
    sensis[:, y_has_sensi] /= sensi_per_y[y_has_sensi]

    # the weight of a sumstat is the sum of the senitivities over all parameters
    # info_weights_red = np.sum(sensis, axis=1)

    m = sensis

    n_in, n_out = m.shape

    np.random.seed(3)

    source = []
    target = []
    value = []
    link_color = []
    node_label = [
        "y1", "y2", *[f"y3_{i+1}" for i in range(4)], "y4", *[f"y5_{i+1}" for i in range(10)],
        *[f"\u03b8{j+1}^{i+1}" for i in range(4) for j in range(4)]]
    all_colors = [
        # sumstats
        "rgba(229,115,115,255)", "rgba(240,98,146,255)", "rgba(186,104,200,255)",
        "rgba(121,134,203,255)", "rgba(100,181,246,255)",
        # parameters
        "rgba(77,182,172,255)", "rgba(129,199,132,255)", "rgba(174,213,129,255)",
        "rgba(220,231,117,255)",
    ]
    node_color = [
        all_colors[0], all_colors[1], *[all_colors[2]] * 4, all_colors[3], *[all_colors[4]] * 10,
        *all_colors[5:] * 4,
        #*[all_colors[5]] * 4, *[all_colors[6]] * 4, *[all_colors[7]] * 4, *[all_colors[8]] * 4,
    ]
    node_x = [*[0 for _ in range(n_in)], *[1 for _ in range(n_out)]]
    node_y = [*[i for i in range(n_in)], *[i for i in range(n_out)]]
    for i_in in range(n_in):
        for i_out in range(n_out):
            source.append(i_in)
            target.append(n_in + i_out)
            value.append(m[i_in, i_out])
            alpha = "0.3"
            if i_in in [0, 1]:
                col = all_colors[i_in]
            elif i_in in range(2, 6):
                col = all_colors[2]
            elif i_in == 6:
                col = all_colors[3]
            else:
                col = all_colors[4]
            link_color.append(col[:-4] + alpha + ")")

    fig = go.Figure(
        data=[
            go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = node_label,
                    color = node_color,
                    #x=node_x,
                    #y=node_y,
                ),
                link = dict(
                    source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target = target,
                    value = value,
                    color = link_color,
                ),
            ),
        ],
    )

    fig.update_layout(
        title_text=f"Data-parameter sensitivities using {predictor_label}",
        title_x=0.5,
        font_size=12,
        width=500, height=900,
        template="simple_white",
    )
    fig.show()

    for fmt in ["png", "pdf"]:
        fig.write_image(f"figures_learn/figure_demo_sankey_{s_predictor}.{fmt}")
