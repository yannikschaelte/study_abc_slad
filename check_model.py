import argparse
import numpy as np
from scipy import spatial
import pyabc
from pyabc.predictor import *
from pyabc.sumstat import *
from pyabc.sumstat.util import read_sample

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str)
args = parser.parse_args()
db = args.db

h = pyabc.History("sqlite:///" + db, create=False)
for t in range(h.max_t):
    pop = h.get_population(t=t)
    sample = pyabc.Sample.from_population(pop)
    par_keys = list(h.get_distribution()[0].columns)
    pre = IdentitySumstat()
    pre.initialize(
        t=0, get_sample=lambda: sample,
        x_0=h.observed_sum_stat(), total_sims=0)
    #print(par_keys)
    x, y, w = read_sample(
        sample=sample, sumstat=pre,
        all_particles=False, par_keys=par_keys)

    subsetter = GMMSubsetter()
    x, y, w = subsetter.select(x, y, w)

    x = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    y = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)

    # print(x.shape, y.shape, w.shape)

    predictor = LinearPredictor()
    predictor.fit(x=x, y=y, w=w)
    y_pred = predictor.predict(x)
    score = ((y - y_pred)**2).sum(axis=0) / x.shape[0]
    score2 = np.abs(y - y_pred).sum(axis=0) / x.shape[0]
    cs_sims = []
    for _y, _y_pred in zip(y.T, y_pred.T):
        # print(_y.shape, _y_pred.shape)
        cs_sims.append((_y * _y_pred).sum() / np.sqrt((_y**2).sum()) / np.sqrt((_y_pred**2).sum()))
    print(score, score2)
