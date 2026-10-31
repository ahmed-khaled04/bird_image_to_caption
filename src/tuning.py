"""Tuning pipeline: grid search and random search on the same space and budget, then a comparison table and plots,
then the best setting retrained fully and compared with the untuned run."""
import json
import math
from pathlib import Path

import pandas as pd

from src.training.search import compare_searches, grid_trials, random_trials, run_search
from src.transfer import comparison_table, run_and_save
from src.visualization import tuning_plots

# settings every trial shares; the search space adds lr, weight_decay and dropout
RUN_KEYS = ["encoder", "freeze", "epochs", "batch_size", "image_size", "optimizer", "seed", "num_workers"]


def search_space(config: dict) -> dict:
    """The grid and random ranges for the configured encoder."""
    if config["encoder"] not in config["spaces"]:
        raise ValueError(f"no search space for encoder '{config['encoder']}' in the config")
    return config["spaces"][config["encoder"]]


def base_run(config: dict) -> dict:
    return {"name": f"{config['encoder']}_tuning_trial", **{key: config[key] for key in RUN_KEYS}}


def make_objective(config: dict, train_fn, checkpoint_dir: Path = None):
    """Score on validation only; shared training automatically saves every trial's best weights."""
    def objective(params: dict) -> dict:
        result = train_fn({**base_run(config), **params}, test=False, checkpoint_dir=checkpoint_dir)
        return {"val_accuracy": result["best_val_accuracy"], "best_epoch": result["best_epoch"],
                "checkpoint_path": result["checkpoint_path"]}
    return objective


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


def best_params(config: dict, table_dir: Path) -> dict:
    """The best setting found by either search, read from the comparison table of `tune`."""
    path = table_dir / f"tuning_{config['encoder']}_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.cli tune` first")
    comparison = pd.read_csv(path, index_col="search")
    winner = comparison[f"best_{config['metric']}"].idxmax()
    params = {p: comparison.loc[winner, f"best_{p}"].item() for p in search_space(config)["grid"]}
    print(f"best setting, found by {winner} search: {params}")
    return params


def retrain_best(config: dict, train_fn, table_dir: Path, checkpoint_dir: Path = None) -> pd.DataFrame:
    """Train the best setting for as many epochs as the transfer runs and score it on validation. It is saved next to
    the transfer runs, so the untuned run of the same encoder and mode is the baseline it is compared with."""
    encoder = config["encoder"]
    run = {**base_run(config), **best_params(config, table_dir), "epochs": config["final_epochs"],
           "name": f"{encoder}_tuned"}
    runs_dir = table_dir / "transfer_runs"
    tuned = run_and_save(run, train_fn, runs_dir, checkpoint_dir)

    baseline_name = "scratch" if encoder == "scratch" else f"{encoder}_{'frozen' if config['freeze'] else 'finetune'}"
    baseline_path = runs_dir / f"{baseline_name}.json"
    if not baseline_path.exists():
        print(f"no untuned baseline yet: run `python -m src.cli transfer --runs {baseline_name}` to compare")
        return pd.DataFrame()
    table = comparison_table([json.loads(baseline_path.read_text()), tuned], [baseline_name])
    table.to_csv(table_dir / f"tuning_{encoder}_baseline_vs_tuned.csv")
    print("\n=== untuned vs tuned")
    print(table.round(4).T.to_string())
    return table
