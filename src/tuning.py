"""Tuning pipeline: grid search and random search on the same space and budget, then a comparison table and plots."""
import math
from pathlib import Path

import pandas as pd

from src.training.search import compare_searches, grid_trials, random_trials, run_search
from src.visualization import tuning_plots


def search_space(config: dict) -> dict:
    """The grid and random ranges for the configured encoder."""
    if config["encoder"] not in config["spaces"]:
        raise ValueError(f"no search space for encoder '{config['encoder']}' in the config")
    return config["spaces"][config["encoder"]]


def synthetic_objective(params: dict) -> dict:
    """Stand-in for training, to test the search pipeline in seconds. Best at lr 1.5e-4, weight decay 1e-3,
    dropout 0.35 (between the grid values), where it returns 0.8; it falls off smoothly around that point."""
    distance = ((math.log10(params["lr"]) - math.log10(1.5e-4)) / 0.5) ** 2 \
        + ((math.log10(params["weight_decay"]) + 3) / 1.5) ** 2 \
        + ((params["dropout"] - 0.35) / 0.3) ** 2
    return {"val_accuracy": 0.8 * math.exp(-distance)}


def run_tuning(config: dict, objective, table_dir: Path, fig_dir: Path) -> pd.DataFrame:
    space, metric, prefix = search_space(config), config["metric"], f"tuning_{config['encoder']}"
    grid = grid_trials(space["grid"])
    # same budget for both, so the comparison is fair
    randoms = random_trials(space["random"], config.get("random_trials") or len(grid), config["seed"])

    results = pd.concat([
        run_search("grid", grid, objective, table_dir / f"{prefix}_grid.csv"),
        run_search("random", randoms, objective, table_dir / f"{prefix}_random.csv"),
    ], ignore_index=True)

    param_names = list(space["grid"])
    comparison = compare_searches(results, metric, param_names)
    comparison.to_csv(table_dir / f"{prefix}_comparison.csv")
    print("\n=== grid search vs random search")
    print(comparison.T.to_string())

    tuning_plots.plot_best_so_far(results, metric, fig_dir / f"{prefix}_best_so_far.png")
    tuning_plots.plot_score_by_param(results, metric, param_names, fig_dir / f"{prefix}_score_by_param.png")
    return comparison
