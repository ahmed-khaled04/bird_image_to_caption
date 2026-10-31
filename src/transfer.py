"""Transfer learning pipeline: train configured runs and compare validation results.

Each finished run is saved as its own JSON file (settings, scores and per-epoch history). A restarted command
skips the runs that already have a file, and runs done on different machines (Colab, Kaggle) are merged by
copying their files into the same folder.
"""
import json
from pathlib import Path

import pandas as pd

from src.training.search import same_params
from src.visualization import transfer_plots


def run_configs(config: dict, names: list = None) -> list:
    """Every run with the defaults filled in; only the runs in `names` if given."""
    runs = [{**config["defaults"], **run} for run in config["runs"]]
    if names:
        unknown = set(names) - {run["name"] for run in runs}
        if unknown:
            raise ValueError(f"unknown runs {sorted(unknown)}, choose from {[run['name'] for run in runs]}")
        runs = [run for run in runs if run["name"] in names]
    return runs


def run_and_save(run: dict, train_fn, runs_dir: Path, checkpoint_dir: Path = None) -> dict:
    path = runs_dir / f"{run['name']}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["run"].keys() != run.keys() or not same_params(saved["run"], run):
            raise ValueError(f"{path} was trained with other settings than {run['name']} has now; "
                             "delete it to train it again")
        print(f"{run['name']}: already done, delete {path} to train it again")
        return saved
    result = train_fn(run, test=False, checkpoint_dir=checkpoint_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float))  # default: numpy numbers are written as floats
    return result


def load_results(runs_dir: Path) -> list:
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("*.json"))]


def comparison_table(results: list, order: list = ()) -> pd.DataFrame:
    """One row per run, in config order, then any other runs (such as the tuned model) by name."""
    rows = [{
        "run": r["run"]["name"], "encoder": r["run"]["encoder"], "mode": r["mode"],
        "trainable_params_M": r["trainable_params"] / 1e6, "total_params_M": r["total_params"] / 1e6,
        "lr": r["run"]["lr"], "epochs": r["run"]["epochs"], "best_epoch": r["best_epoch"],
        "best_val_accuracy": r["best_val_accuracy"],
        "minutes": r["minutes"], "minutes_per_epoch": r["minutes"] / len(r["history"]),
    } for r in results]
    rank = {name: i for i, name in enumerate(order)}
    rows.sort(key=lambda row: (rank.get(row["run"], len(rank)), row["run"]))
    return pd.DataFrame(rows).set_index("run")


def report(config: dict, table_dir: Path, fig_dir: Path, runs_dir: Path) -> pd.DataFrame:
    results = load_results(runs_dir)
    if not results:
        print("no finished runs yet")
        return pd.DataFrame()
    table = comparison_table(results, [run["name"] for run in config["runs"]])
    table.to_csv(table_dir / "transfer_comparison.csv")
    print("\n=== transfer learning vs scratch (best validation accuracy)")
    print(table.round(4).T.to_string())
    results.sort(key=lambda r: list(table.index).index(r["run"]["name"]))  # plot in table order
    transfer_plots.plot_learning_curves(results, fig_dir / "transfer_learning_curves.png")
    transfer_plots.plot_validation_scores(table, fig_dir / "transfer_validation_scores.png")
    return table


def run_transfer(config: dict, train_fn, table_dir: Path, fig_dir: Path, checkpoint_dir: Path = None,
                 names: list = None) -> pd.DataFrame:
    runs_dir = table_dir / "transfer_runs"
    for run in run_configs(config, names):
        run_and_save(run, train_fn, runs_dir, checkpoint_dir)
    return report(config, table_dir, fig_dir, runs_dir)
