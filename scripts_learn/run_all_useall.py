"""Main study execution script.

Compare:
* Unweighted vs adaptive
* 

"""

import os
import numpy as np
import logging

import pyabc
from pyabc import ABCSMC, RedisEvalParallelSampler
from pyabc.distance import *
from pyabc.sumstat import *
from pyabc.predictor import *
from pyabc.util import *

import slad
from slad.util import *
from slad.ana_util import *


# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

# read cmd line arguments
host, port = slad.read_args()

# constants
sim_frac = 0.4

def get_distance(
    name: str,
    pre,
    total_sims: int,
) -> pyabc.Distance:
    #if name == "PNorm":
    #    return PNormDistance(p=1)

    par_trafos = [lambda x: x, lambda x: x**2]

    mlp_unn = MLPPredictor(
        normalize_features=False,
        normalize_labels=False,
        hidden_layer_sizes=HiddenLayerHandle(method="mean"),
        solver="adam",
        max_iter=20000,
        early_stopping=True,
    )

    mlp = MLPPredictor(
        hidden_layer_sizes=HiddenLayerHandle(method="mean"),
        solver="adam",
        max_iter=20000,
        early_stopping=True,
    )

    if name == "Adaptive":
        return AdaptivePNormDistance(p=1, scale_function=mad)

    # Linear

    if name == "Linear__Initial":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(normalize_features=False, normalize_labels=False),
                fit_ixs={0},
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Linear":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                LinearPredictor(normalize_features=False, normalize_labels=False),
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Adaptive__Linear":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                LinearPredictor(),
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Adaptive__Linear__Extend":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                LinearPredictor(),
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                par_trafo=ParTrafo(trafos=par_trafos),
                all_particles=True,
            ),
        )

    # MLP

    if name == "MLP2__Initial":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=mlp_unn,
                fit_ixs={0},
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "MLP2":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=mlp_unn,
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Adaptive__MLP2":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=mlp,
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Adaptive__MLP2__Extend":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=mlp,
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                par_trafo=ParTrafo(trafos=par_trafos),
                all_particles=True,
            ),
        )

    # MS

    if name == "MS2__Initial":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(normalize_features=False, normalize_labels=False),
                        mlp_unn,
                    ],
                ),
                fit_ixs={0},
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "MS2":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(normalize_features=False, normalize_labels=False),
                        mlp_unn,
                    ],
                ),
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Adaptive__MS2":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(),
                        mlp,
                    ],
                ),
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                all_particles=True,
            ),
        )

    if name == "Adaptive__MS2__Extend":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(),
                        mlp,
                    ],
                ),
                fit_ixs=EventIxs(sims=[sim_frac * total_sims]),
                pre=pre,
                par_trafo=ParTrafo(trafos=par_trafos),
            ),
        )

    # Info Linear

    if name == "Info__Linear__Initial":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs={0},
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__Linear":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=EventIxs(sims=[sim_frac * total_sims]),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__Linear__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=EventIxs(sims=[sim_frac * total_sims]),
            par_trafo=ParTrafo(trafos=par_trafos),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    # Info MLP

    if name == "Info__MLP2__Initial":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs={0},
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__MLP2":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=[sim_frac * total_sims]),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__MLP2__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=[sim_frac * total_sims]),
            par_trafo=ParTrafo(trafos=par_trafos),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )
    
    # Info MS

    if name == "Info__MS2__Initial":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
            ),
            fit_info_ixs={0},
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__MS2":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
            ),
            fit_info_ixs=EventIxs(sims=[sim_frac * total_sims]),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__MS2__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
            ),
            fit_info_ixs=EventIxs(sims=[sim_frac * total_sims]),
            par_trafo=ParTrafo(trafos=par_trafos),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    raise ValueError(f"Distance {name} not recognized.")


distance_names = [
    "Adaptive",
    # Linear
    "Linear__Initial",
    "Linear",
    "Adaptive__Linear",
    "Adaptive__Linear__Extend",
    # MLP
    "MLP2__Initial",
    "MLP2",
    "Adaptive__MLP2",
    "Adaptive__MLP2__Extend",
    # MS
    #"MS2__Initial",
    #"MS2",
    #"Adaptive__MS2",
    #"Adaptive__MS2__Extend",
    # Info Linear
    "Info__Linear__Initial",
    "Info__Linear",
    "Info__Linear__Extend",
    # Info MLP
    "Info__MLP2__Initial",
    "Info__MLP2",
    "Info__MLP2__Extend",
    # Info MS
    #"Info__MS2__Initial",
    #"Info__MS2",
    #"Info__MS2__Extend",
]

# test
for distance_name in distance_names:
    get_distance(distance_name, pre=pyabc.IdentitySumstat(), total_sims=10000)


n_rep = 10
data_dir_base = "data_learn_useall"

problem_types = [
    #"demo",
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
        data_dir = os.path.join(data_dir_base, f"{problem.get_id()}_{i_rep}")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            data = problem.get_obs()
            save_data(data, data_dir)


# main loop
for problem_type in problem_types:
    print(problem_type)

    for i_rep in range(n_rep):
        problem = type_to_problem(problem_type)
        pop_size, max_total_sim = n_sim_for_problem(problem_type)

        # output folder
        data_dir = os.path.join(data_dir_base, f"{problem.get_id()}_{i_rep}")

        # get data
        data = load_data(problem, data_dir)

        # iterate over distances
        for dname in distance_names:
            print(problem, i_rep, dname)
            db_file = os.path.join(data_dir, f"db_{dname}.db")
            if os.path.exists(db_file):
                print(f"{db_file} exists already, continuing with next")
                continue

            # define distance
            distance = get_distance(
                dname,
                #re=problem.get_sumstat(),
                pre=None,
                total_sims=max_total_sim,
            )
            if isinstance(distance, AdaptivePNormDistance):
                distance.scale_log_file = os.path.join(
                    data_dir, f"log_scale_{dname}.json"
                )
            if isinstance(distance, InfoWeightedPNormDistance):
                distance.info_log_file = os.path.join(
                    data_dir, f"log_info_{dname}.json"
                )

            sampler = RedisEvalParallelSampler(host=host, port=port, batch_size=10)
            abc = ABCSMC(
                problem.get_model(),
                problem.get_prior(),
                distance,
                sampler=sampler,
                population_size=pop_size,
            )
            abc.new(db="sqlite:///" + db_file, observed_sum_stat=data)
            abc.run(max_total_nr_simulations=max_total_sim)


print("ABC out")
