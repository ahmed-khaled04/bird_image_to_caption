"""Evaluate saved best-epoch checkpoints without retraining the models."""
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from torch import nn

from src.data.loader import class_names
from src.data.loaders import build_dataloaders
from src.evaluation.classification import classification_tables, evaluate
from src.experiment import pick_device
from src.training.checkpoints import load_checkpoint
from src.training.checkpoint_refs import read_checkpoint, resolve_checkpoint, verify_class_labels
from src.visualization.evaluation_plots import plot_top_confusions


def load_model(run, checkpoint_dir, device, saved=None, source_path=None, names=None):
    path = resolve_checkpoint(saved or {"run": run}, checkpoint_dir, source_path)
    if names is not None:
        verify_class_labels(read_checkpoint(path), names)
    model, _ = load_checkpoint(path, device)
    return model


def confusion_pairs(confusion):
    pairs = confusion.stack().rename("count").reset_index()
    pairs = pairs[(pairs["true_species"] != pairs["predicted_species"]) & (pairs["count"] > 0)]
    return pairs.sort_values("count", ascending=False).reset_index(drop=True)


def save_details(result, names, prefix, table_dir, fig_dir):
    scores, per_class, confusion = classification_tables(result, names)
    table_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(table_dir / f"{prefix}_per_class.csv", index=False)
    confusion.to_csv(table_dir / f"{prefix}_confusion.csv")
    pairs = confusion_pairs(confusion)
    pairs.to_csv(table_dir / f"{prefix}_top_confusions.csv", index=False)
    plot_top_confusions(pairs, fig_dir / f"{prefix}_top_confusions.png", prefix.replace("_", " "))
    return scores


def evaluate_run(saved, split, data, names, checkpoint_dir, table_dir, fig_dir, device, num_workers,
                 source_path=None):
    run = saved["run"]
    model = load_model(run, checkpoint_dir, device, saved, source_path, names)
    loader = build_dataloaders(
        {split: data[split]}, image_size=run["image_size"], batch_size=run["batch_size"],
        num_workers=num_workers, seed=run["seed"]
    )[split]
    result = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    scores = save_details(result, names, f"{split}_{run['name']}", table_dir, fig_dir)
    del model
    return scores


def run_evaluation(data_dir: Path, runs_dir: Path, checkpoint_dir: Path, table_dir: Path,
                   fig_dir: Path, selected_runs=None, test_run=None, num_workers=0, device="auto"):
    paths = sorted(runs_dir.glob("*.json"))
    saved = {path.stem: json.loads(path.read_text()) for path in paths}
    if selected_runs:
        missing = set(selected_runs) - saved.keys()
        if missing:
            raise ValueError(f"no saved runs for {sorted(missing)}")
        saved = {name: saved[name] for name in selected_runs}
    if not saved:
        raise ValueError(f"no saved runs in {runs_dir}")
    if test_run and test_run not in saved:
        raise ValueError(f"--test-run {test_run} must be among the evaluated runs")

    # Check every selected reference before doing any expensive evaluation or writing reports.
    references = {name: resolve_checkpoint(result, checkpoint_dir, runs_dir / f"{name}.json")
                  for name, result in saved.items()}
    data = load_from_disk(str(data_dir))
    names = class_names(data["validation"])
    device = pick_device() if device == "auto" else torch.device(device)
    rows = []
    for name, run_result in saved.items():
        scores = evaluate_run(run_result, "validation", data, names, checkpoint_dir, table_dir,
                              fig_dir, device, num_workers, runs_dir / f"{name}.json")
        row = {
            "run": name, "encoder": run_result["run"]["encoder"], "mode": run_result["mode"],
            "best_epoch": run_result["best_epoch"], "images": scores["images"],
            "accuracy": scores["accuracy"], "top5_accuracy": scores["top5_accuracy"],
            "precision_macro": scores["precision_macro"], "recall_macro": scores["recall_macro"],
            "f1_macro": scores["f1_macro"], "loss": scores["loss"],
            "trainable_params_M": run_result["trainable_params"] / 1e6,
            "total_params_M": run_result["total_params"] / 1e6,
            "train_minutes": run_result["minutes"],
            "checkpoint_path": str(references[name]),
        }
        rows.append(row)
        print(f"validated {name}: accuracy {scores['accuracy']:.3f}, macro F1 {scores['f1_macro']:.3f}")

    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False).set_index("run")
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / "validation_comparison.csv")
    print("\n=== validation comparison (all saved best-epoch checkpoints)")
    print(table[["encoder", "mode", "best_epoch", "accuracy", "top5_accuracy",
                 "precision_macro", "recall_macro", "f1_macro", "train_minutes"]].round(3).to_string())

    if test_run:
        winner = saved[test_run]
        scores = evaluate_run(winner, "test", data, names, checkpoint_dir, table_dir,
                              fig_dir, device, num_workers, runs_dir / f"{test_run}.json")
        final = pd.DataFrame([{"run": test_run, "validation_accuracy": table.loc[test_run, "accuracy"],
                               **{f"test_{key}": value for key, value in scores.items()}}]).set_index("run")
        final.to_csv(table_dir / "final_test_metrics.csv")
        print("\n=== final test result")
        print(final.round(3).to_string())
    provenance = {"data_dir": str(data_dir), "runs_dir": str(runs_dir), "device": str(device),
                  "test_run": test_run, "checkpoints": {name: str(path) for name, path in references.items()}}
    (table_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return table
