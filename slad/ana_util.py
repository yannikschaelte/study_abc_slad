import slad


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


def n_sim_for_problem(problem_type):
    """Get population size and max. total simulations for problem type."""
    if problem_type == "demo":
        pop_size = 4000
        max_total_sim = 1000000
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
        pop_size = 500 #200
        max_total_sim = 125000 #50000
    elif problem_type == "fearnhead_gk":
        pop_size = 1000
        max_total_sim = 250000
    elif problem_type == "fearnhead_lv":
        pop_size = 500 #200
        max_total_sim = 125000 #50000
    elif problem_type == "harrison_toy":
        pop_size = 2000
        max_total_sim = 40000
    else:
        raise ValueError(f"Problem not recognized: {problem_type}")

    return pop_size, max_total_sim
