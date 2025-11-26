"""Strategy configurations used for benchmarking / selection."""

# Each label maps to kwargs for TSPSolver.solve.
STRATEGIES: dict[str, dict] = {
    "nn": {"method": "nearest_neighbor"},
    "nn_2opt": {"method": "nearest_neighbor", "apply_two_opt_flag": True},
    "nn_anneal": {
        "method": "nearest_neighbor",
        "apply_two_opt_flag": True,
        "annealing": True,
    },
    "greedy_2opt": {"method": "greedy", "apply_two_opt_flag": True},
    "greedy_anneal": {
        "method": "greedy",
        "apply_two_opt_flag": True,
        "annealing": True,
    },
    "random_ga": {
        "method": "random",
        "apply_two_opt_flag": True,
        "genetic": True,
    },
}
