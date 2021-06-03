import os
import numpy as np
import matplotlib.pyplot as plt

import slad


def load_data(data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


dir = os.path.dirname(os.path.realpath(__file__))


for problem in [
    slad.PrangleLVErrorProblem(n_obs_error=0),
    slad.PrangleLVErrorProblem(),
]:
    data_dir = os.path.join(dir, "..", "data_robust", problem.get_id())
    data = load_data(data_dir)

    fig, ax = plt.subplots()
    ax.plot(data["y"][:, 0])
    ax.plot(data["y"][:, 1])
    plt.savefig(f"plot_robust/data_{problem.get_id()}.png")
