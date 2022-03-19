import os
import sys
import numpy as np
import logging

import pyabc
from pyabc import ABCSMC, RedisEvalParallelSampler
from pyabc.distance import *
from pyabc.predictor import *
from pyabc.sumstat import *
from pyabc.util import *

import slad

# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

# read cmd line arguments
host, port = slad.read_args()


def get_distance(name: str) -> pyabc.Distance:
    # ms2 is without heuristic and adam instead
    mlp = MLPPredictor(
        hidden_layer_sizes=HiddenLayerHandle(method="mean"),
        solver="adam",
        max_iter=20000,
        early_stopping=True,
    )

    if name == "Euclidean":
        return PNormDistance(p=2)
    if name == "Manhattan":
        return PNormDistance(p=1)

    if name == "Calibrated__Euclidean__mad":
        return AdaptivePNormDistance(p=2, scale_function=mad, fit_scale_ixs={0})
    if name == "Calibrated__Manhattan__mad":
        return AdaptivePNormDistance(p=1, scale_function=mad, fit_scale_ixs={0})

    if name == "Adaptive__Euclidean__mad":
        return AdaptivePNormDistance(p=2, scale_function=mad)
    if name == "Adaptive__Manhattan__mad":
        return AdaptivePNormDistance(p=1, scale_function=mad)

    if name == "Adaptive__Euclidean__cmad":
        return AdaptivePNormDistance(p=2, scale_function=cmad)
    if name == "Adaptive__Manhattan__cmad":
        return AdaptivePNormDistance(p=1, scale_function=cmad)

    if name == "Adaptive__Euclidean__mad_or_cmad":
        return AdaptivePNormDistance(p=2, scale_function=mad_or_cmad)
    if name == "Adaptive__Manhattan__mad_or_cmad":
        return AdaptivePNormDistance(p=1, scale_function=mad_or_cmad)

    if name == "Adaptive__Linear__Manhattan__mad_or_cmad__useall":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(
                    normalize_features=False, normalize_labels=False,
                ),
                fit_ixs=EventIxs(sims=0.4 * 150000),
                pre_before_fit=False,
                all_particles=True,
                # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            ),
        )

    if name == "Adaptive__Linear__Manhattan__mad_or_cmad__Extend__useall":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            sumstat=PredictorSumstat(
                predictor=LinearPredictor(
                    normalize_features=False, normalize_labels=False,
                ),
                fit_ixs=EventIxs(sims=0.4 * 150000),
                pre_before_fit=False,
                par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
                # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
                all_particles=True,
            ),
        )

    if name == "Adaptive__MS2__Manhattan__mad_or_cmad":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            sumstat=PredictorSumstat(
                predictor=ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(),
                        mlp,
                    ],
                    split_method="cross_validation",
                ),
                fit_ixs=EventIxs(sims=0.4 * 150000),
                pre_before_fit=False,
                # par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
                # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            ),
        )

    if name == "Adaptive__MS2__Manhattan__mad_or_cmad__Extend":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            sumstat=PredictorSumstat(
                predictor=ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(),
                        mlp,
                    ],
                    split_method="cross_validation",
                ),
                fit_ixs=EventIxs(sims=0.4 * 150000),
                pre_before_fit=False,
                par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
                # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            ),
        )

    if name == "Adaptive__MLP2__Manhattan__mad_or_cmad__useall":
        return AdaptivePNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            sumstat=PredictorSumstat(
                predictor=mlp,
                fit_ixs=EventIxs(sims=0.4 * 150000),
                pre_before_fit=False,
                #par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
                # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            ),
        )

    if name == "Info__Linear__Manhattan__mad_or_cmad__useall":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            predictor=LinearPredictor(
                normalize_features=False, normalize_labels=False,
            ),
            fit_info_ixs=EventIxs(sims=0.4 * 150000),
            feature_normalization="mad",
            all_particles_for_prediction=True,
            # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
        )

    if name == "Info__Linear__Manhattan__mad_or_cmad__Subset":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            predictor=LinearPredictor(
                normalize_features=False, normalize_labels=False,
            ),
            fit_info_ixs=EventIxs(sims=0.4 * 150000),
            feature_normalization="mad",
            subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
        )

    if name == "Info__Linear__Manhattan__mad_or_cmad__Extend__useall":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            predictor=LinearPredictor(
                normalize_features=False, normalize_labels=False,
            ),
            fit_info_ixs=EventIxs(sims=0.4 * 150000),
            feature_normalization="mad",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            all_particles_for_prediction=True,
        )

    if name == "Info__MS2__Manhattan__mad_or_cmad":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
                split_method="cross_validation",
            ),
            fit_info_ixs=EventIxs(sims=0.4 * 150000),
            feature_normalization="mad",
            # par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
        )

    if name == "Info__MS2__Manhattan__mad_or_cmad__Extend__useall":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    mlp,
                ],
                split_method="cross_validation",
            ),
            fit_info_ixs=EventIxs(sims=0.4 * 150000),
            feature_normalization="mad",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            all_particles_for_prediction=True,
        )

    if name == "Info__MLP2__Manhattan__mad_or_cmad__Extend__useall":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad_or_cmad,
            predictor=mlp,
            fit_info_ixs=EventIxs(sims=0.4 * 150000),
            feature_normalization="mad",
            par_trafo=ParTrafo(trafos=[lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]),
            # subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 10}),
            all_particles_for_prediction=True,
        )

    raise ValueError(f"Distance {name} not recognized.")


distance_names = [
    #"Euclidean",
    #"Manhattan",
    "Calibrated__Euclidean__mad",
    #"Calibrated__Manhattan__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Manhattan__mad",
    #"Adaptive__Euclidean__cmad",
    #"Adaptive__Manhattan__cmad",
    # "Adaptive__Euclidean__mad_or_cmad",
    "Adaptive__Manhattan__mad_or_cmad",
    #"Info__Linear__Manhattan__mad_or_cmad",
    "Adaptive__Linear__Manhattan__mad_or_cmad__useall",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__useall",
    #"Adaptive__MS__Manhattan__mad_or_cmad",
    #"Adaptive__MS2__Manhattan__mad_or_cmad__Extend",
    "Info__Linear__Manhattan__mad_or_cmad__useall",
    #"Info__Linear__Manhattan__mad_or_cmad__Subset",
    #"Info__MS__Manhattan__mad_or_cmad",
    "Info__MS2__Manhattan__mad_or_cmad__Extend__useall",
    "Info__MLP2__Manhattan__mad_or_cmad__Extend__useall",
    "Info__Linear__Manhattan__mad_or_cmad__Extend__useall",
    "Adaptive__Linear__Manhattan__mad_or_cmad__Extend__useall",
]

# test
for distance_name in distance_names:
    get_distance(distance_name)


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


n_rep = 1

# create data
for i_rep in range(n_rep):
    problem = slad.TumorErrorProblem(noisy=True, frac_error=0)

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")
    if os.path.exists(data_dir):
        data = load_data(problem, data_dir)
    else:
        os.makedirs(data_dir)
        data = problem.get_obs()
        save_data(data, data_dir)

    # errored data
    problem = slad.TumorErrorProblem(noisy=True)

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")
    if os.path.exists(data_dir):
        continue
    os.makedirs(data_dir)

    data = problem.errorfy(data)
    save_data(data, data_dir)

for i_rep in range(n_rep):

    for kwargs in [
        {"noisy": True, "frac_error": 0},
        {"noisy": True, "frac_error": 0.1},
    ]:
        problem = slad.TumorErrorProblem(**kwargs)
        pop_size = 500
        max_total_sim = 150000
        #pop_size = 1000
        #max_total_sim = 250000
        #pop_size = 200
        #max_total_sim = 50000

        model = problem.get_model()
        prior = problem.get_prior()
        gt_par = problem.get_gt_par()

        # output folder
        dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dir, "..", "data_robust", f"{problem.get_id()}_{i_rep}")

        # get data
        data = load_data(problem, data_dir)

        for distance_name in distance_names:
            print(kwargs, distance_name)

            db_file = os.path.join(data_dir, f"db_{distance_name}.db")
            if os.path.exists(db_file):
                print(f"{db_file} exists already, continuing with next")
                continue

            distance = get_distance(distance_name)
            if isinstance(distance, AdaptivePNormDistance):
                distance.scale_log_file = os.path.join(
                    data_dir, f"log_scale_{distance_name}.json"
                )
            if isinstance(distance, InfoWeightedPNormDistance):
                distance.info_log_file = os.path.join(
                    data_dir, f"log_info_{distance_name}.json"
                )

            sampler = RedisEvalParallelSampler(host=host, port=port, batch_size=1)
            abc = ABCSMC(
                model, prior, distance, sampler=sampler, population_size=pop_size
            )
            abc.new(db="sqlite:///" + db_file, observed_sum_stat=data)
            abc.run(max_total_nr_simulations=max_total_sim)

            print(f"ABC out {kwargs} {distance_name}")
            raise ValueError("Done!!!!")
            sys.exit(0)
            #return
