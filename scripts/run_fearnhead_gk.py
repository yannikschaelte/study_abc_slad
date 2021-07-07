import os
import numpy as np
import logging

import slad
import pyabc
from pyabc import ABCSMC, RedisEvalParallelSampler
from pyabc.distance import *
from pyabc.sumstat import *
from pyabc.predictor import *

# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

# read cmd line arguments
host, port = slad.read_args()

# load problem
problem = slad.FearnheadGKProblem()
model = problem.get_model()
prior = problem.get_prior()
gt_par = problem.get_gt_par()
id = problem.get_id()

# output folder
dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir, "..", "data", id)
os.makedirs(data_dir, exist_ok=True)

# variable definitions
n_rep = 3
pop_size = 1000
max_total_sim = 100000

# load or generate and save data
if os.path.exists(os.path.join(data_dir, f"data_{problem.get_y_keys()[0]}.csv")):
    data = {}
    for key in problem.get_y_keys():
        data[key] = np.loadtxt(os.path.join(data_dir, f"data_{key}.csv"), delimiter=",")
else:
    # simulate
    data = problem.get_obs()
    for key, val in data.items():
        if not isinstance(val, np.ndarray):
            val = np.array([val])
        np.savetxt(os.path.join(data_dir, f"data_{key}.csv"), val, delimiter=",")

for i_rep in range(n_rep):
    distances = {
        "Euclidean": PNormDistance(),
        "Calibrated": AdaptivePNormDistance(fit_scale_ixs=1),
        # for scale function comparison
        "Adaptive": AdaptivePNormDistance(),
        "Euclidean_Linear": PNormDistance(
            sumstat=PredictorSumstat(
                LinearPredictor(normalize_features=True, normalize_labels=False),
                normalize_labels=False,
            ),
        ),
        "Adaptive_Linear_initial": AdaptivePNormDistance(
            sumstat=PredictorSumstat(LinearPredictor(), fit_ixs={0}),
        ),
        "Adaptive_Linear": AdaptivePNormDistance(
            sumstat=PredictorSumstat(LinearPredictor()),
        ),
        "Adaptive_MS": AdaptivePNormDistance(
            sumstat=PredictorSumstat(
                ModelSelectionPredictor(
                    predictors=[
                        LinearPredictor(),
                        GPPredictor(GPKernelHandle(ard=False)),
                        MLPPredictor(),
                        # MLPPredictor(hidden_layer_sizes=HiddenLayerHandle("mean")),
                    ],
                )
            ),
        ),
        "Adaptive_MLP_Initial_Many": AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(MLPPredictor(
                hidden_layer_sizes=(50, 50, 50), solver="adam", max_iter=500),
                fit_ixs={0}),
        ),
        # info distances
        "Info_MS": InfoWeightedPNormDistance(
            predictor=ModelSelectionPredictor(
                predictors=[
                    LinearPredictor(),
                    MLPPredictor(),
                    # MLPPredictor(hidden_layer_sizes=HiddenLayerHandle("mean")),
                ],
            ),
        ),
        "Info_Linear": InfoWeightedPNormDistance(predictor=LinearPredictor()),
    }

    for distance_label, distance in distances.items():
        db_file = os.path.join(data_dir, f"db_{distance_label}_{i_rep}.db")
        if os.path.exists(db_file):
            print(f"{db_file} exists already, continuing with next")
            continue
        if isinstance(distance, AdaptivePNormDistance):
            distance.scale_log_file = os.path.join(
                data_dir, f"log_scale_{distance_label}_{i_rep}.json"
            )
        if isinstance(distance, InfoWeightedPNormDistance):
            distance.info_log_file = os.path.join(
                data_dir, f"log_info_{distance_label}_{i_rep}.json"
            )

        if "MLP_Initial_Many" in distance_label:
            population_size = pyabc.ConstantPopulationSize(nr_particles=pop_size, nr_calibration_particles=100000)
            max_total_nr_simulations = max_total_sim + 100000
        else:
            population_size = pop_size
            max_total_nr_simulations = max_total_sim

        sampler = RedisEvalParallelSampler(host=host, port=port, batch_size=10)
        abc = ABCSMC(
            model,
            prior,
            distance,
            sampler=sampler,
            population_size=population_size,
        )
        abc.new(db="sqlite:///" + db_file, observed_sum_stat=data)
        abc.run(
            max_total_nr_simulations=max_total_nr_simulations,
        )

print(f"ABC {id} out")
