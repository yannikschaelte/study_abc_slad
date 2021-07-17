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


def get_distance(name: str, pre, total_sims) -> pyabc.Distance:
    if name == "Adaptive" or name == "Adaptive__Long":
        return AdaptivePNormDistance(p=1, scale_function=mad)

    if name == "Linear__Initial":
        return PNormDistance(
            p=1, sumstat=PredictorSumstat(LinearPredictor(), fit_ixs={0}, pre=pre),
        )

    if name == "Adaptive__Linear__Initial":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(LinearPredictor(), fit_ixs={0}, pre=pre),
        )

    if name == "Adaptive__Linear" of name == "Adaptive__Linear__Long":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                LinearPredictor(),
                fit_ixs=pyabc.EventIxs(total_sims=[0.4 * total_sims, 0.6 * total_sims]),
                pre=pre),
        )

    if name == "GP__Initial":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                GPPredictor(kernel=GPKernelHandle(ard=False)),
                fit_ixs={0},
            ),
        )

    if name == "Adaptive__GP__Initial":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                GPPredictor(kernel=GPKernelHandle(ard=False)),
                fit_ixs={0},
            ),
        )

    if name == "Adaptive__GP":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                GPPredictor(kernel=GPKernelHandle(ard=False)),
                fit_ixs=pyabc.EventIxs(total_sims=[0.4 * total_sims, 0.6 * total_sims]),
            ),
        )

    if name == "Adaptive__GP__Subset":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                GPPredictor(kernel=GPKernelHandle(ard=False)),
                fit_ixs=pyabc.EventIxs(total_sims=[0.4 * total_sims, 0.6 * total_sims]),
                subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 20}),
            ),
        )

    if name == "Adaptive__GP__Initial__Unnorm":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                GPPredictor(kernel=GPKernelHandle(ard=False), normalize_features=False, normalize_labels=False),
                fit_ixs=pyabc.EventIxs(total_sims=[0.4 * total_sims, 0.6 * total_sims]),
                subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 20}),
            ),
        )

    if name == "MLP__Initial":
        return PNormDistance(
            p=1,
            sumstat=PredictorSumstat(
                MLPPredictor(
                    hidden_layer_sizes=HiddenLayerHandle(method="max", n_layer=3),
                    solver="adam",
                    max_iter=200,
                ),
                fit_ixs={0},
            ),
        )

    if name == "Adaptive__MLP__Initial":
        return AdaptivePNormDistance(
            p=1, scale_function=mad,
            sumstat=PredictorSumstat(
                MLPPredictor(
                    hidden_layer_sizes=HiddenLayerHandle(method="max", n_layer=3),
                    solver="adam",
                    max_iter=200,
                ),
                fit_ixs={0},
            ),
        )

    if name == "Info__Linear__Initial":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs={0},
        )

    if name == "Info__Linear":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=pyabc.EventIxs(total_sims=[0.3 * total_sims, 0.5 * total_sims, 0.7 * total_sims]),
        )

    if name == "Info__Linear__mad":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=pyabc.EventIxs(total_sims=[0.3 * total_sims, 0.5 * total_sims, 0.7 * total_sims]),
            feature_normalization="mad",
        )

    if name == "Info__Linear__Subset":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=pyabc.EventIxs(total_sims=[0.3 * total_sims, 0.5 * total_sims, 0.7 * total_sims]),
            subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 20}),
        )

    if name == "Info__Linear__Subset__mad" or name == "Info__Linear__Subset__mad__Long":
        return InfoWeightedPNormDistance(
            p=1,
            scale_function=mad,
            predictor=LinearPredictor(),
            fit_info_ixs=pyabc.EventIxs(total_sims=[0.3 * total_sims, 0.5 * total_sims, 0.7 * total_sims]),
            subsetter=GMMSubsetter(gmm_args={"max_iter": 1000, "n_init": 20}),
            feature_normalization="mad",
        )

    raise ValueError(f"Distance {name} not recognized.")


distance_names = [
    "Adaptive",
    "Adaptive__Long",
    "Linear__Initial",
    "Adaptive__Linear__Initial",
    "Adaptive__Linear",
    "Adaptive__Linear__Long",
    "Adaptive__Linear__Cheat",
    "GP__Initial",
    "Adaptive__GP__Initial",
    "Adaptive__GP",
    "Adaptive__GP__Subset",
    "Adaptive__GP__Initial__Unnorm",
    #"MLP__Initial",
    #"Adaptive__MLP__Initial",
    "Info__Linear__Initial",
    "Info__Linear",
    "Info__Linear__mad",
    "Info__Linear__Subset",
    "Info__Linear__Subset__mad",
    "Info__Linear__Subset__mad__Long",
]

# test
for distance_name in distance_names:
    if distance_name != "Adaptive__Linear__Cheat":
        get_distance(distance_name, pre=pyabc.IdentitySumstat(), total_sims=10000)


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


n_rep = 5


def type_to_problem(problem_type, **kwargs):
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
    raise ValueError("Problem not recognized")


problem_types = [
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
]


# create data
for problem_type in problem_types:
    for i_rep in range(n_rep):
        problem = type_to_problem(problem_type)

        dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dir, "..", "data_learn", f"{problem.get_id()}_{i_rep}")
        if os.path.exists(data_dir):
            data = load_data(problem, data_dir)
        else:
            os.makedirs(data_dir)
            data = problem.get_obs()
            save_data(data, data_dir)


for problem_type in problem_types:
    print(problem_type)

    problem = type_to_problem(problem_type)

    for i_rep in range(n_rep):
        problem = type_to_problem(problem_type)

        if problem_type == "prangle_normal":
            pop_size = 1000
            max_total_sim = 50000
        elif problem_type == "prangle_gk":
            pop_size = 1000
            max_total_sim = 500000
        elif problem_type == "prangle_lv":
            pop_size = 500
            max_total_sim = 200000
        elif problem_type == "fearnhead_gk":
            pop_size = 1000
            max_total_sim = 1000000
        elif problem_type == "fearnhead_lv":
            pop_size = 500
            max_total_sim = 200000
        else:
            raise ValueError("Problem not recognized")

        model = problem.get_model()
        prior = problem.get_prior()
        gt_par = problem.get_gt_par()
        pre = problem.get_sumstat()

        # output folder
        dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(
            dir, "..", "data_learn", f"{problem.get_id()}_{i_rep}"
        )

        # get data
        data = load_data(problem, data_dir)
        
        _max_total_sim = max_total_sim

        for distance_name in distance_names:
            print(distance_name)
            if "Long" in distance_name and "gk" in problem_type:
                max_total_sim = _max_total_sim * 3

            sampler = RedisEvalParallelSampler(host=host, port=port, batch_size=10)

            if distance_name == "Adaptive__Linear__Cheat":
                cheat_db_file = os.path.join(data_dir, f"db_Cheat.db")
                if not os.path.exists(cheat_db_file):
                    h = pyabc.History(
                        "sqlite:///" + os.path.join(data_dir, "db_Adaptive.db"), create=False)
                    df, w = h.get_distribution()
                    bounds = {}
                    for key in problem.get_gt_par().keys():
                        bounds[key] = (pyabc.weighted_quantile(df[key].to_numpy(), w, alpha=0.025),
                                       pyabc.weighted_quantile(df[key].to_numpy(), w, alpha=0.975))
                    cheat_prior = pyabc.Distribution(**{key: pyabc.RV("uniform", lb, ub-lb)
                                                        for key, (lb, ub) in bounds.items()})
                    abc = ABCSMC(
                        model, cheat_prior, PNormDistance(), sampler=sampler, population_size=population_size,
                        eps=pyabc.ListEpsilon([np.inf]),
                    )
                    abc.new(db="sqlite:///" + cheat_db_file, observed_sum_stat=data)
                    abc.run(max_nr_populations=1)
                h = pyabc.History("sqlite:///" + cheat_db_file, create=False)
                sample = pyabc.Sample.from_population(h.get_population())
                sumstat = PredictorSumstat(LinearPredictor(), fit_ixs={0}, pre=pre)
                sumstat.initialize(t=0, get_sample=lambda: sample, x_0=data, total_sims=0)
                sumstat.fit_ixs.ts = []
                distance = AdaptivePNormDistance(
                    p=1, scale_function=mad,
                    sumstat=sumstat,
                )
            else:
                distance = get_distance(distance_name, pre=pre, total_sims=max_total_sim)

            nr_calibration_particles = pop_size
            if "GP__Initial" in distance_name or "MLP__Initial" in distance_name:
                nr_calibration_particles = pop_size
                # nr_calibration_particles = int(0.2 * max_total_sim)
            population_size = pyabc.ConstantPopulationSize(
                nr_particles=pop_size,
                nr_calibration_particles=nr_calibration_particles,
            )

            db_file = os.path.join(data_dir, f"db_{distance_name}.db")
            if os.path.exists(db_file):
                print(f"{db_file} exists already, continuing with next")
                continue

            if isinstance(distance, AdaptivePNormDistance):
                distance.scale_log_file = os.path.join(
                    data_dir, f"log_scale_{distance_name}.json"
                )
            if isinstance(distance, InfoWeightedPNormDistance):
                distance.info_log_file = os.path.join(
                    data_dir, f"log_info_{distance_name}.json"
                )

            abc = ABCSMC(
                model, prior, distance, sampler=sampler, population_size=population_size
            )
            abc.new(db="sqlite:///" + db_file, observed_sum_stat=data)
            abc.run(max_total_nr_simulations=max_total_sim)

print("ABC out")
