import pyabc
import slad

problem_types = [
    "prangle_normal",
    "prangle_gk",
    "prangle_lv",
    "fearnhead_gk",
    "fearnhead_lv",
]

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


for problem_type in problem_types:
    print(problem_type)
    problem = type_to_problem(problem_type)
    h = pyabc.History(
        f"sqlite:///data_learn/{problem.get_id()}_{0}/db_Adaptive.db", create=False)
    df, w = h.get_distribution()
    for key in problem.get_gt_par().keys():
        print(key, min(df[key].to_numpy()), max(df[key].to_numpy()),
              pyabc.weighted_quantile(df[key].to_numpy(), w, alpha=0.025),
              pyabc.weighted_quantile(df[key].to_numpy(), w, alpha=0.975))
