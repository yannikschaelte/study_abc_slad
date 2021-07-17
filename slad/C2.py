"""Constant definitions."""

distance_names = [
    "Adaptive",
    "Adaptive__Linear__Cheat",
    "Linear__Initial",
    "Adaptive__Linear__Initial",
    "Adaptive__Linear",
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
]

distance_labels = {dname: dname for dname in distance_names}
distance_labels_short = {dname: dname for dname in distance_names}
distance_colors = {dname: f"C{i}" for i, dname in enumerate(distance_names)}

problem_labels = {
    "prangle_normal": "Prangle-Normal",
    "prangle_gk": "Prangle-GK",
    "prangle_lv": "Prangle-LV",
    "fearnhead_gk": "Fearnhead-GK",
    "fearnhead_lv": "Fearnhead-LV",
    "uninf": "Uninformative",
    "gaussian": "Replicates",
    "gk": "GK",
    "lv": "Lotka-Volterra",
    "cr-zero": "Conversion",
}

parameter_labels = {
    "prangle_normal": {"theta": r"$\theta$"},
    "prangle_gk": {"A": "A", "B": "B", "g": "g", "k": "k"},
    "prangle_lv": {
        "p1": r"$\log~\theta_1$",
        "p2": r"$\log~\theta_2$",
        "p3": r"$\log~\theta_3$",
    },
    "fearnhead_gk": {"A": "A", "B": "B", "g": "g", "k": "k"},
    "fearnhead_lv": {
        "p1": r"$\log~\theta_1$",
        "p2": r"$\log~\theta_2$",
        "p3": r"$\log~\theta_3$",
    },
    "uninf": {"p0": r"$\theta$"},
    "gaussian": {"p0": r"$\theta$"},
    "gk": {"A": "A", "B": "B", "g": "g", "k": "k"},
    "lv": {
        "p1": r"$\log~\theta_1$",
        "p2": r"$\log~\theta_2$",
        "p3": r"$\log~\theta_3$",
    },
    "cr-zero": {"p0": r"$\log~\theta_1$", "p1": r"$\log~\theta_2$"},
    "tumor": {
        "log_division_rate": "log(div. rate)",
        "log_division_depth": "log(div. depth)",
        "log_initial_spheroid_radius": "log(init. sph. rad.)",
        "log_initial_quiescent_cell_fraction": "log(init. quies. cell frac.)",
        "log_ecm_production_rate": "log(ECM prod. rate)",
        "log_ecm_degradation_rate": "log(ECM degrad. rate)",
        "log_ecm_division_threshold": "log(ECM div. thresh.)",
    },
}

data_xlabels = {
    "tumor": {
        "growth_curve": "Time [$d$]",
        "proliferation_profile": "Distance to rim [$10^{-5} m$]",
        "extra_cellular_matrix_profile": "Distance to rim [$10^{-5} m$]",
    },
    "uninf": {"y": "Coordinate"},
    "gaussian": {"y": "Replicate"},
    "gk": {"y": "Order statistic"},
    "lv": {"Prey": "Time [au]", "Predator": "Time [au]"},
    "cr-zero": {"y": "Time [au]"},
    "cr-swap": {"y": "Time [au]"},
}

data_ylabels = {
    "uninf": {"y": "Value"},
    "gaussian": {"y": "Value"},
    "gk": {"y": "Value"},
    "lv": {"Prey": "Number", "Predator": "Number"},
    "cr-zero": {"y": "Species B"},
    "cr-swap": {"y": "Species B"},
    "tumor": {
        "growth_curve": "Spheroid radius [$\\mu m$]",
        "proliferation_profile": "Frac. proliferating cells",
        "extra_cellular_matrix_profile": "ECM intensity",
    },
}
