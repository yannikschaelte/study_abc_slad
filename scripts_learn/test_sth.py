import os
import pyabc
import slad

base_dir = "data_learn/fearnhead_gk_10000_100_0/"
base_dir = "data_learn/prangle_gk_0/"

problem = slad.FearnheadGKProblem()
gt_par = problem.get_gt_par()

for f in os.listdir(base_dir):
    if not f.endswith(".db"):
        continue
    print(f)
    h = pyabc.History("sqlite:///" + base_dir + f, create=False)
    df, w = h.get_distribution()
    for key in gt_par:
        print(key, pyabc.weighted_mse(df[key].to_numpy(), w, refval=gt_par[key]))
