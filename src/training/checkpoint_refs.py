"""Resolve recorded weights without confusing reruns with historical experiments."""
import math
from pathlib import Path

import torch

from src.training.checkpoints import MODELS_DIR, PROJECT_ROOT


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"run", "state_dict"} <= payload.keys():
        raise ValueError(f"{path} must contain run and state_dict")
    return payload


def verify_checkpoint(payload, saved, path="checkpoint"):
    """Allow enriched reconstruction metadata, but require every saved setting to match."""
    actual = payload["run"]
    mismatches = [key for key, value in saved["run"].items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"{path}: checkpoint settings differ for {', '.join(mismatches)}")
    if "best_epoch" in saved and "epoch" in payload and saved["best_epoch"] != payload["epoch"]:
        raise ValueError(f"{path}: best epoch does not match the saved experiment")
    score = payload.get("best_val_accuracy", payload.get("validation_metric"))
    if score is not None and "best_val_accuracy" in saved and not math.isclose(
            score, saved["best_val_accuracy"], rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{path}: best validation accuracy does not match the saved experiment")


def resolve_checkpoint(saved, checkpoint_dir=None, source_path=None):
    """Resolve an explicit reference (including a relocated basename), or a legacy run file.

    Never infer an original run's weights from matching hyperparameters or scores.
    An explicit but missing reference does not fall back to a different run-name file.
    """
    roots = [Path(checkpoint_dir)] if checkpoint_dir is not None else []
    roots += [MODELS_DIR, PROJECT_ROOT / "outputs" / "checkpoints"]
    reference = saved.get("checkpoint_path")
    if reference:
        recorded = Path(reference).expanduser()
        candidates = [recorded] if recorded.is_absolute() else [PROJECT_ROOT / recorded]
        if source_path is not None and not recorded.is_absolute():
            candidates.append(Path(source_path).resolve().parent / recorded)
        candidates += [root / recorded.name for root in roots]
    else:
        name = saved["run"]["name"]
        if Path(name).name != name:
            raise ValueError("Run name must be a filename, not a path")
        candidates = [root / f"{name}{suffix}" for root in roots for suffix in (".pt", ".pth")]
    found = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if not found:
        raise FileNotFoundError(
            f"No checkpoint for {saved['run']['name']}. Recover the original weights and record "
            "checkpoint_path in its result JSON; a new training run must have a separate result record.")
    if len(found) > 1:
        # An existing explicit path disambiguates copies. Otherwise require the user to choose.
        if reference and candidates[0].is_file():
            found = [candidates[0].resolve()]
        else:
            raise ValueError(f"Multiple checkpoints for {saved['run']['name']}: {found}; record an explicit path")
    path = found[0]
    verify_checkpoint(read_checkpoint(path), saved, path)
    return path


def verify_class_labels(payload, names):
    count = payload["state_dict"]["fc.weight"].shape[0]
    if len(names) != count:
        raise ValueError(f"Expected {count} class labels, got {len(names)}")
    if payload.get("class_labels") is not None and list(payload["class_labels"]) != list(names):
        raise ValueError("Checkpoint class labels differ from dataset labels or their index order")
