"""Run demonstration example.

* Unweighted
* (Calibrated)
* Adaptive
* Linear
* Adaptive linear
* Adaptive linear subset
* Info linear
* Info linear subset
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
pilot_frac = 0.4
n_rep = 1
data_dir_base = "data_learn_demo_useall"


def get_distance(
    name: str,
    pre,
    total_sims: int,
) -> pyabc.Distance:
    sims = {pilot_frac * total_sims}

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

    if name == "PNorm":
        return PNormDistance(p=1)

    if name == "Adaptive":
        return AdaptivePNormDistance(p=1, scale_function=mad)

    if name == "Linear":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(
                    normalize_features=False, normalize_labels=False,
                ),
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                all_particles=True,
            ),
        )

    if name == "Adaptive__Linear":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(),
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                all_particles=True,
            ),
        )

    if name == "Adaptive__Linear__Extend":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(),
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
                all_particles=True,
            ),
        )

    if name == "MLP2":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                predictor=mlp_unn,
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                all_particles=True,
            ),
        )

    if name == "Adaptive__MLP2":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=mlp,
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                all_particles=True,
            ),
        )

    if name == "Adaptive__MLP2__Extend":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad,
            sumstat=PredictorSumstat(
                predictor=mlp,
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
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
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
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
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
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
                fit_ixs=EventIxs(sims=sims),
                pre=pre,
                pre_before_fit=False,
                par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
                all_particles=True,
            ),
        )

    if name == "Info__Linear":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__Linear__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
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
            fit_info_ixs=EventIxs(sims=sims),
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
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            all_particles_for_prediction=True,
        )

    if name == "Info__MLP2":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            all_particles_for_prediction=True,
        )

    if name == "Info__MLP2__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            all_particles_for_prediction=True,
        )
    
    # info without parameter normalization

    if name == "Info2__Linear":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            normalize_by_par=False,
            all_particles_for_prediction=True,
        )

    if name == "Info2__Linear__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            normalize_by_par=False,
            all_particles_for_prediction=True,
        )

    if name == "Info2__MS2":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
            ),
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            normalize_by_par=False,
            all_particles_for_prediction=True,
        )

    if name == "Info2__MS2__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
            ),
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            normalize_by_par=False,
            all_particles_for_prediction=True,
        )

    if name == "Info2__MLP2":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            normalize_by_par=False,
            all_particles_for_prediction=True,
        )

    if name == "Info2__MLP2__Extend":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=sims),
            feature_normalization="weights",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            normalize_by_par=False,
            all_particles_for_prediction=True,
        )

    raise ValueError(f"Distance {name} not recognized.")


distance_names = [
    "PNorm",
    "Adaptive",
    "Linear",
    "Adaptive__Linear",
    "Adaptive__Linear__Extend",
    "MLP2",
    "Adaptive__MLP2",
    "Adaptive__MLP2__Extend",
    #"MS2",
    #"Adaptive__MS2",
    #"Adaptive__MS2__Extend",
    "Info__Linear",
    "Info__Linear__Extend",
    "Info__MLP2",
    "Info__MLP2__Extend",
    #"Info__MS2",
    #"Info__MS2__Extend",
    "Info2__Linear",
    "Info2__Linear__Extend",
    "Info2__MLP2",
    "Info2__MLP2__Extend",
    #"Info2__MS2",
    #"Info2__MS2__Extend",
]


# test
for dname in distance_names:
    get_distance(dname, pre=IdentitySumstat(), total_sims=1000)

# problem types
problem_types = [
    "demo",
]

# create data
for problem_type in problem_types:
    for i_rep in range(n_rep):
        problem = type_to_problem(problem_type)
        data_dir = os.path.join(data_dir_base, f"{problem.get_id()}_{i_rep}")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            data = problem.get_obs()
            data = {"s": np.array([0, 0, *[0] * 4, 0.7**2, *[0] * 10])}
            save_data(data, data_dir)


# main loop
for problem_type in problem_types:
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
                pre=problem.get_sumstat(),
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
                distance.info_sample_log_file = os.path.join(
                    data_dir, f"log_info_sample_{dname}"
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
