"""Constant definitions."""

distance_names = [
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Euclidean__cmad",
    "Adaptive__Euclidean__mad_or_cmad",
    "Calibrated__Manhattan__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__cmad",
    "Adaptive__Manhattan__mad_or_cmad",
]


distance_labels = {
    "Calibrated__Euclidean__mad": "L2 + Cal. + MAD",
    "Adaptive__Euclidean__mad": "L2 + Adap. + MAD",
    "Adaptive__Euclidean__cmad": "L2 + Adap. + CMAD",
    "Adaptive__Euclidean__mad_or_cmad": "L2 + Adap. + (C)MAD",
    "Calibrated__Manhattan__mad": "L1 + Cal. + MAD",
    "Adaptive__Manhattan__mad": "L1 + Adap. + MAD",
    "Adaptive__Manhattan__cmad": "L1 + Adap. + CMAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD",
}

distance_labels_short = {
    "Calibrated__Euclidean__mad": "L2+Cal.+MAD",
    "Adaptive__Euclidean__mad": "L2+Ada.+MAD",
    "Adaptive__Euclidean__cmad": "L2+Ada.+CMAD",
    "Adaptive__Euclidean__mad_or_cmad": "L2+Ada.+PCMAD",
    "Calibrated__Manhattan__mad": "L1+Cal.+MAD",
    "Adaptive__Manhattan__mad": "L1+Ada.+MAD",
    "Adaptive__Manhattan__cmad": "L1+Ada.+CMAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1+Ada.+PCMAD",
}

distance_colors = {
    "Calibrated__Euclidean__mad": "C3",
    "Adaptive__Euclidean__mad": "C2",
    "Adaptive__Euclidean__cmad": "grey",
    "Adaptive__Euclidean__mad_or_cmad": "grey",
    "Calibrated__Manhattan__mad": "grey",
    "Adaptive__Manhattan__mad": "C1",
    "Adaptive__Manhattan__cmad": "grey",
    "Adaptive__Manhattan__mad_or_cmad": "C0",
    "Adaptive__Linear__Manhattan__mad_or_cmad": "C1",
    "Adaptive__Linear__Manhattan__mad_or_cmad__Extend": "C2",
    "Adaptive__MS__Manhattan__mad_or_cmad__Extend": "C7",
    "Info__Linear__Manhattan__mad_or_cmad": "C3",
    "Info__Linear__Manhattan__mad_or_cmad__Subset": "grey",
    "Info__Linear__Manhattan__mad_or_cmad__Extend": "C5",
    "Info__MS__Manhattan__mad_or_cmad__Extend": "C6",
}

distance_labels_short_learn = {
    "Adaptive": "L1+Ada.+MAD",
    "Linear__Initial": "L1+StatLR+Init",
    "Linear": "L1+StatLR",
    "Adaptive__Linear": "L1+Ada.+MAD+StatLR",
    "Adaptive__Linear__Extend": "L1+Ada.+MAD+StatLR+P4",
    "MLP2__Initial": "L1+StatNN+Init",
    "MLP2": "L1+StatNN",
    "Adaptive__MLP2": "L1+Ada.+MAD+StatNN",
    "Adaptive__MLP2__Extend": "L1+Ada.+MAD+StatNN+P4",
    "MS2__Initial": "L1+StatMS+Init",
    "MS2": "L1+StatMS",
    "Adaptive__MS2": "L1+Ada.+MAD+StatMS",
    "Adaptive__MS2__Extend": "L1+Ada.+MAD+StatMS+P4",
    # Info
    "Info__Linear__Initial": "L1+Ada.+MAD+SensiLR+Init",
    "Info__Linear": "L1+Ada.+MAD+SensiLR",
    "Info__Linear__Extend": "L1+Ada.+MAD+SensiLR+P4",
    "Info__MLP2__Initial": "L1+Ada.+MAD+SensiNN+Init",
    "Info__MLP2": "L1+Ada.+MAD+SensiNN",
    "Info__MLP2__Extend": "L1+Ada.+MAD+SensiNN+P4",
    "Info__MS2__Initial": "L1+Ada.+MAD+SensiMS+Init",
    "Info__MS2": "L1+Ada.+MAD+SensiMS",
    "Info__MS2__Extend": "L1+Ada.+MAD+SensiMS+P4",
    # tumor
    "Adaptive__Manhattan__mad_or_cmad": "L1+Ada.+PCMAD",
    "Adaptive__Linear__Manhattan__mad_or_cmad__useall": "L1+Ada.+PCMAD+StatLR",
    "Adaptive__Linear__Manhattan__mad_or_cmad__Extend__useall": "L1+Ada.+PCMAD+StatLR+P4",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__useall": "L1+Ada.+PCMAD+StatNN",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__Extend__useall": "L1+Ada.+PCMAD+StatNN+P4",
    "Info__Linear__Manhattan__mad_or_cmad__useall": "L1+Ada.+PCMAD+SensiLR",
    "Info__Linear__Manhattan__mad_or_cmad__Extend__useall": "L1+Ada.+PCMAD+SensiLR+P4",
    "Info__MLP2__Manhattan__mad_or_cmad__useall": "L1+Ada.+PCMAD+SensiNN",
    "Info__MLP2__Manhattan__mad_or_cmad__Extend__useall": "L1+Ada.+PCMAD+SensiNN+P4",
    # stuff
    "Info2__Linear__Initial": "L1+Ada.+MAD+SensiUnnLR+Init",
    "Info2__Linear": "L1+Ada.+MAD+SensiUnnLR",
    "Info2__Linear__Extend": "L1+Ada.+MAD+SensiUnnLR+P4",
    "Info2__MLP2__Initial": "L1+Ada.+MAD+SensiUnnNN+Init",
    "Info2__MLP2": "L1+Ada.+MAD+SensiUnnNN",
    "Info2__MLP2__Extend": "L1+Ada.+MAD+SensiUnnNN+P4",
    "Info2__MS2__Initial": "L1+Ada.+MAD+SensiUnnMS+Init",
    "Info2__MS2": "L1+Ada.+MAD+SensiUnnMS",
    "Info2__MS2__Extend": "L1+Ada.+MAD+SensiUnnMS+P4",
}

distance_colors_learn = {
    "Adaptive": "C5",
    "Linear__Initial": "grey",
    "Linear": "C4",
    "Adaptive__Linear": "C3",
    "Adaptive__Linear__Extend": "C1",
    "MLP2__Initial": "grey",
    "MLP2": "C4",
    "Adaptive__MLP2": "C3",
    "Adaptive__MLP2__Extend": "C1",
    "MS2__Initial": "grey",
    "MS2": "C4",
    "Adaptive__MS2": "C3",
    "Adaptive__MS2__Extend": "C1",
    "Info__Linear__Initial": "grey",
    "Info__Linear": "C2",
    "Info__Linear__Extend": "C0",
    "Info__MLP2__Initial": "grey",
    "Info__MLP2": "C2",
    "Info__MLP2__Extend": "C0",
    "Info__MS2__Initial": "grey",
    "Info__MS2": "C2",
    "Info__MS2__Extend": "C0",
    # tumor
    "Adaptive__Manhattan__mad_or_cmad": "C5",
    "Adaptive__Linear__Manhattan__mad_or_cmad__useall": "C3",
    "Adaptive__Linear__Manhattan__mad_or_cmad__Extend__useall": "C1",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__useall": "C3",#"C6",
    "Adaptive__MLP2__Manhattan__mad_or_cmad__Extend__useall": "C1",
    "Info__Linear__Manhattan__mad_or_cmad__useall": "C2",
    "Info__Linear__Manhattan__mad_or_cmad__Extend__useall": "C0",
    "Info__MLP2__Manhattan__mad_or_cmad__useall": "C2",
    "Info__MLP2__Manhattan__mad_or_cmad__Extend__useall": "C0",#"C4",
}

# non-grey colors
distance_colors_learn_sep = {
    "Adaptive": "C5",
    "Linear": "C4",
    "Adaptive__Linear": "C3",
    "Adaptive__Linear__Extend": "C1",
    "MLP2": "C4",
    "Adaptive__MLP2": "C3",
    "Adaptive__MLP2__Extend": "C1",
    "MS2": "C4",
    "Adaptive__MS2": "C3",
    "Adaptive__MS2__Extend": "C1",
    "Info__Linear": "C2",
    "Info__Linear__Extend": "C0",
    "Info__MLP2": "C2",
    "Info__MLP2__Extend": "C0",
    "Info__MS2": "C2",
    "Info__MS2__Extend": "C0",
    # stuff
    "Info2__Linear": "C7",
    "Info2__Linear__Extend": "C6",
    "Info2__MLP2": "C7",
    "Info2__MLP2__Extend": "C6",
    "Info2__MS2": "C7",
    "Info2__MS2__Extend": "C6",
}

problem_labels_old = {
    "uninf": "Uninformative",
    "gaussian": "Replicates",
    "gk": "GK",
    "lv": "Lotka-Volterra",
    "cr-zero": "Conversion",
}

problem_labels = {
    "uninf": "M1",
    "gaussian": "M2",
    "cr-zero": "M3",
    "gk": "M4",
    "lv": "M5",
    "tumor": "M6",
}

problem_labels_learn = {
    "cr": "T1",
    "prangle_normal": "T2",
    "prangle_gk": "T3",
    "prangle_lv": "T4",
    "fearnhead_gk": "T5",
    "fearnhead_lv": "T6",
    "harrison_toy": "T7",
    "tumor": "Tumor",
}

parameter_labels = {
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
    # learn
    "demo": {
        "p1": r"$\theta_1$", "p2": r"$\theta_2$",
        "p3": r"$\theta_3$", "p4": r"$\theta_4$",
    },
    "cr": {"p0": r"$\log~\theta_1$", "p1": r"$\log~\theta_2$"},
    "prangle_normal": {"theta": r"$\theta$"},
    "prangle_gk": {"A": "A", "B": "B", "g": "g", "k": "k"},
    "prangle_lv": {
        "p1": r"$\log~\theta_1$",
        "p2": r"$\log~\theta_2$",
        "p3": r"$\log~\theta_3$",
    },
    "fearnhead_gk": {"A": "A", "B": "B", "g": "g", "k": "k"},
    "fearnhead_lv": {
        "p1": r"$\theta_1$",
        "p2": r"$\theta_2$",
        "p3": r"$\theta_3$",
    },
    "harrison_toy": {"theta": r"$\theta$"},
}

data_xlabels = {
    "tumor": {
        "growth_curve": "Time [$d$]",
        "proliferation_profile": "Distance to rim [x$10^{-5} m$]",
        "extra_cellular_matrix_profile": "Distance to rim [x$10^{-5} m$]",
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
