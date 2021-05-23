import os
import numpy as np

import slad
from pyabc import ABCSMC, RedisEvalParallelSampler
from pyabc.distance import *
from pyabc.sumstat import *
from pyabc.predictor import *


# read cmd line arguments
host, port = slad.read_args()
print(f"Python script with host={host}, port={port}")

# load problem
problem = slad.CoreProblem()
model = problem.get_model()
prior = problem.get_prior()
gt_par = problem.get_gt_par()
id = problem.get_id()

# output folder
dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir, "..", "data", id)
os.makedirs(data_dir, exist_ok=True)

# variable definitions
n_rep = 10
pop_size = 1000
max_total_sim = 100000

# generate and save data
data = problem.get_obs()
for key, val in data.items():
    np.savetxt(os.path.join(data_dir, f"data_{key}.csv"), val, delimiter=",")

for i_rep in range(n_rep):
    distances = {
        "Euclidean": PNormDistance(),
        "Calibrated": AdaptivePNormDistance(fit_scale_ixs=1),
        # for scale function comparison
        "Adaptive_std": AdaptivePNormDistance(scale_function=std),
        "Adaptive": AdaptivePNormDistance(),
        "Euclidean_Linear": PNormDistance(
            sumstat=PredictorSumstat(
                LinearPredictor(normalize_features=False, normalize_labels=False),
                normalize_labels=False,
            ),
        ),
        "Linear_initial": AdaptivePNormDistance(
            sumstat=PredictorSumstat(LinearPredictor(), fit_ixs={0}),
        ),
        "Linear": AdaptivePNormDistance(
            sumstat=PredictorSumstat(LinearPredictor()),
        ),
        "Linear_Subset": AdaptivePNormDistance(
            sumstat=PredictorSumstat(LinearPredictor(), subsetter=GMMSubsetter())
        ),
        "MS": AdaptivePNormDistance(
            sumstat=PredictorSumstat(
                ModelSelectionPredictor(
                    predictors=[LinearPredictor(), GPPredictor(), MLPPredictor()],
                )
            ),
        ),
        "MS_Subset": AdaptivePNormDistance(
            sumstat=PredictorSumstat(
                ModelSelectionPredictor(
                    predictors=[LinearPredictor(), GPPredictor(), MLPPredictor()],
                ),
                subsetter=GMMSubsetter(),
            ),
        ),
        # info distances
        "Info": InfoWeightedPNormDistance(
            predictor=ModelSelectionPredictor(
                predictors=[LinearPredictor(), MLPPredictor()],
            ),
        ),
        "Info_Subset": InfoWeightedPNormDistance(
            predictor=ModelSelectionPredictor(
                predictors=[LinearPredictor(), MLPPredictor()],
            ),
            subsetter=GMMSubsetter(),
        ),
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

        sampler = RedisEvalParallelSampler(host=host, port=port, batch_size=10)
        abc = ABCSMC(
            model,
            prior,
            distance,
            sampler=sampler,
            population_size=pop_size,
        )
        abc.new(db="sqlite:///" + db_file, observed_sum_stat=data)
        abc.run(
            max_total_nr_simulations=max_total_sim,
        )
