"""Hyperparameter search: grid search and random search, compared on the same budget.

A search only calls `objective(params) -> dict of scores`. It does not know how a model is trained, so the
same code tunes any model. Every finished trial is written to a CSV at once, and a restarted search skips the
trials already in it, so a dropped Colab session loses at most the trial that was running.
"""
import itertools
import math
import random
import time
from pathlib import Path

import pandas as pd


def grid_trials(grid: dict) -> list:
    """Every combination of the listed values: {"lr": [1e-4, 1e-3], "dropout": [0.2, 0.5]} gives 4 trials."""
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*grid.values())]


def sample(spec: dict, rng: random.Random):
    """One value from {"log_uniform": [low, high]}, {"uniform": [low, high]} or {"choice": [a, b, ...]}.

    Log-uniform gives every factor of 10 the same chance, which suits the learning rate and weight decay:
    1e-4 to 1e-3 matters as much as 1e-3 to 1e-2.
    """
    (kind, values), = spec.items()
    if kind == "log_uniform":
        return math.exp(rng.uniform(math.log(values[0]), math.log(values[1])))
    if kind == "uniform":
        return rng.uniform(values[0], values[1])
    if kind == "choice":
        return rng.choice(values)
    raise ValueError(f"unknown range '{kind}', use log_uniform, uniform or choice")


def random_trials(space: dict, n_trials: int, seed: int) -> list:
    """`n_trials` random settings, all drawn up front from `seed`, so a restarted search draws the same ones."""
    rng = random.Random(seed)
    return [{name: sample(spec, rng) for name, spec in space.items()} for _ in range(n_trials)]


def same_params(row: dict, params: dict) -> bool:
    # values come back from the CSV, so floats are compared with a tolerance
    return all(row[name] == value or (isinstance(value, float) and math.isclose(row[name], value))
               for name, value in params.items())


def run_search(name: str, trials: list, objective, results_path: Path) -> pd.DataFrame:
    """Run `objective` on every trial in order and return one row per trial: its params, scores and seconds."""
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    if len(rows) > len(trials) or not all(same_params(row, params) for row, params in zip(rows, trials)):
        raise ValueError(f"{results_path} holds other trials than this search; delete it to start over")
    if rows:
        print(f"{name}: {len(rows)} of {len(trials)} trials already done, resuming")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    for trial in range(len(rows), len(trials)):
        params = trials[trial]
        start = time.perf_counter()
        scores = objective(params)
        rows.append({"search": name, "trial": trial + 1, **params, **scores, "seconds": time.perf_counter() - start})
        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"{name} {trial + 1}/{len(trials)}  "
              + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in {**params, **scores}.items()))
    return pd.DataFrame(rows)


def compare_searches(results: pd.DataFrame, metric: str, param_names: list) -> pd.DataFrame:
    """One row per search: number of trials, best score and its settings, when it was found and the time taken."""
    rows = []
    for name, runs in results.groupby("search", sort=False):
        runs = runs.reset_index(drop=True)
        best = runs[metric].idxmax()
        rows.append({
            "search": name,
            "trials": len(runs),
            f"best_{metric}": runs.loc[best, metric],
            "best_found_at_trial": best + 1,
            "minutes_to_best": runs.loc[:best, "seconds"].sum() / 60,
            "total_minutes": runs["seconds"].sum() / 60,
            **{f"best_{p}": runs.loc[best, p] for p in param_names},
            **{f"distinct_{p}": runs[p].nunique() for p in param_names},
        })
    return pd.DataFrame(rows).set_index("search")
