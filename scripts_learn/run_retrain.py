"""Impact of the fraction spent in the initial iteration on predictors."""

import os
import numpy as np
import logging

import pyabc
from pyabc import ABCSMC, RedisEvalParallelSampler
from pyabc.distance import *
from pyabc.sumstat import *
from pyabc.predictor import *

import slad


# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

# read cmd line arguments
host, port = slad.read_args()


def get_distance(
    name: str,
    pre,
    total_sims: int,
    pilot_frac: float,
) -> pyabc.Distance:
    if name == "Linear__Subset__Retrain":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(
                    normalize_features=False, normalize_labels=False,
                ),
                fit_ixs=EventIxs(from_sims=pilot_frac * total_sims),
                pre=pre,
                pre_before_fit=False,
                subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            ),
        )

    if name == "Adaptive__Linear__Subset__Retrain":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(
                    normalize_features=False, normalize_labels=False,
                ),
                fit_ixs=EventIxs(from_sims=pilot_frac * total_sims),
                pre=pre,
                pre_before_fit=False,
                subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            ),
        )

    if name == "Info__Linear__Subset__Retrain":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(
                normalize_features=False, normalize_labels=False),
            fit_info_ixs=EventIxs(from_sims=pilot_frac * total_sims),
            feature_normalization="weights",
            subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
        )

    raise ValueError(f"Distance {name} not recognized.")


distance_names = [
    "Linear__Subset__Retrain",
    "Adaptive__Linear__Subset__Retrain",
    "Info__Linear__Subset__Retrain",
]

# test
for distance_name in distance_names:
    get_distance(distance_name, pre=pyabc.IdentitySumstat(), total_sims=10000, pilot_frac=0.1)


def save_data(data, data_dir):
    for key, val in data.items():
        if not isinstance(val, np.ndarray):
            val = np.array([val])
        np.savetxt(os.path.join(data_dir, f"data_{key}.csv"), val, delimiter=",")


def load_data(problem, data_dir):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
        if data[key].size == 1:
            data[key] = float(data[key])
    return data


# fractions spent in no-info mode
fracs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7]
# number of repetitions
n_rep = 3

def type_to_problem(problem_type, **kwargs):
    if problem_type == "demo":
        return slad.DemoProblem(**kwargs)
    if problem_type == "cr":
        return slad.CRProblem(**kwargs)
    if problem_type == "prangle_normal":
        return slad.PrangleNormalProblem(**kwargs)
    if problem_type == "prangle_gk":
        return slad.PrangleGKProblem(**kwargs)
    if problem_type == "prangle_lv":
        return slad.PrangleLVProblem(**kwargs)
    if problem_type == "fearnhead_gk":
        return slad.FearnheadGKProblem(**kwargs)
    if problem_type == "fearnhead_lv":
        return slad.FearnheadLVProblem(**kwargs)
    if problem_type == "harrison_toy":
        return slad.HarrisonToyProblem(**kwargs)
    raise ValueError(f"Problem not recognized: {problem_type}")


problem_types = [
    "demo",
    "cr",
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
    "harrison_toy",
]


# create data
for problem_type in problem_types:
    for i_rep in range(n_rep):
        problem = type_to_problem(problem_type)

        data_dir = os.path.join("data_learn_pilot", f"{problem.get_id()}_{i_rep}")
        if os.path.exists(data_dir):
            data = load_data(problem, data_dir)
        else:
            os.makedirs(data_dir)
            data = problem.get_obs()
            save_data(data, data_dir)


for problem_type in problem_types:
    print(problem_type)

    for i_rep in range(n_rep):
        for frac in fracs:
            problem = type_to_problem(problem_type)

            if problem_type == "demo":
                pop_size = 1000
                max_total_sim = 250000
            elif problem_type == "cr":
                pop_size = 1000
                max_total_sim = 250000
            elif problem_type == "prangle_normal":
                pop_size = 1000
                max_total_sim = 25000
            elif problem_type == "prangle_gk":
                pop_size = 1000
                max_total_sim = 250000
            elif problem_type == "prangle_lv":
                pop_size = 200
                max_total_sim = 50000
            elif problem_type == "fearnhead_gk":
                pop_size = 1000
                max_total_sim = 250000
            elif problem_type == "fearnhead_lv":
                pop_size = 200
                max_total_sim = 50000
            elif problem_type == "harrison_toy":
                pop_size = 2000
                max_total_sim = 40000
            else:
                raise ValueError("Problem not recognized")

            model = problem.get_model()
            prior = problem.get_prior()
            gt_par = problem.get_gt_par()
            pre = problem.get_sumstat()

            # output folder
            data_dir = os.path.join(
                "data_learn_pilot", f"{problem.get_id()}_{i_rep}"
            )

            # get data
            data = load_data(problem, data_dir)

            for distance_name in distance_names:
                print(distance_name)

                db_file = os.path.join(data_dir, f"db_frac_{frac}_{distance_name}.db")
                if os.path.exists(db_file):
                    print(f"{db_file} exists already, continuing with next")
                    continue

                distance = get_distance(distance_name, pre=pre, total_sims=max_total_sim, pilot_frac=frac)

                sampler = RedisEvalParallelSampler(host=host, port=port, batch_size=10)
                abc = ABCSMC(
                    model, prior, distance, sampler=sampler, population_size=pop_size
                )
                abc.new(db="sqlite:///" + db_file, observed_sum_stat=data)
                abc.run(max_total_nr_simulations=max_total_sim)

print("ABC out")
