"""Shared, weights-only checkpoints for all classifier training workflows."""
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def checkpoint_directory(directory=None):
    """Keep all new weights under the project root, independent of the working directory."""
    root = MODELS_DIR.resolve()
    directory = root if directory is None else Path(directory).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"Checkpoints must be stored inside {root}; omit checkpoint_dir to use it")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_default(value):
    if isinstance(value, (Path, torch.device)):
        return str(value)
    # Search configurations can contain NumPy scalar values.
    if hasattr(value, "item") and not isinstance(value, torch.Tensor):
        return value.item()
    raise TypeError(f"Checkpoint metadata must contain plain values, got {type(value).__name__}")


def training_metadata(model, loaders, optimizer, epochs, run_config=None, class_labels=None):
    """Collect reconstruction settings without changing the caller's run configuration."""
    run = {**(run_config or {}), **getattr(model, "checkpoint_config", {})}
    if "encoder" not in run:
        raise ValueError("Build the classifier with build_classifier() or supply its encoder in run_config")
    run.setdefault("name", run["encoder"])
    run.setdefault("num_classes", model.fc.out_features)
    run.setdefault("dropout", model.dropout.p)
    run.setdefault("freeze", bool(getattr(model.encoder, "frozen", False)))
    run["epochs"] = epochs
    run.setdefault("optimizer", type(optimizer).__name__.lower())
    run.setdefault("optimizer_defaults", optimizer.defaults)
    train_loader = loaders["train"]
    run.setdefault("batch_size", getattr(train_loader, "batch_size", None))
    run.setdefault("num_workers", getattr(train_loader, "num_workers", 0))
    run.setdefault("seed", torch.initial_seed())
    # Direct fit() users get the same crop size used by ClassificationDataset.
    dataset = getattr(loaders["validation"], "dataset", None)
    for transform in getattr(getattr(dataset, "transform", None), "transforms", []):
        if type(transform).__name__ == "CenterCrop":
            run.setdefault("image_size", int(transform.size[0]))
    run.setdefault("image_size", 224)
    if class_labels is None:
        dataset = getattr(train_loader, "dataset", None)
        features = getattr(getattr(dataset, "split", None), "features", {})
        class_labels = getattr(features.get("label"), "names", None)
    if class_labels is None:
        class_labels = [str(index) for index in range(run["num_classes"])]
    if len(class_labels) != run["num_classes"]:
        raise ValueError("class_labels must have one entry per model output, in label-index order")
    # JSON conversion ensures weights_only=True needs no custom pickle globals.
    metadata = json.loads(json.dumps({"run": run, "class_labels": list(class_labels)}, default=_json_default))
    return metadata["run"], metadata["class_labels"]


def save_checkpoint(model, run, epoch, validation_metric, class_labels, state_dict=None, directory=None):
    """Publish one immutable, unique .pth only after serialization succeeds.

    Hard-linking the completed temporary file atomically refuses an existing filename.
    Failed writes and interrupted training cannot replace a previous experiment's weights.
    """
    if not math.isfinite(validation_metric):
        raise ValueError("Cannot save a non-finite validation metric")
    directory = checkpoint_directory(directory)
    created = datetime.now(timezone.utc)
    run_id = uuid4().hex
    slug = lambda value: re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:64] or "run"
    filename = (f"{slug(run['encoder'])}_{slug(run['name'])}_"
                f"{created:%Y%m%dT%H%M%S%fZ}_{run_id}_best.pth")
    destination = directory / filename
    state = copy.copy(model.state_dict() if state_dict is None else state_dict)
    for name, tensor in state.items():
        state[name] = tensor.detach().cpu()
    metadata = json.loads(json.dumps({"run": run, "class_labels": class_labels}, default=_json_default))
    checkpoint = {
        "state_dict": state, **metadata, "epoch": int(epoch),
        "validation_metric": float(validation_metric), "metric_name": "val_accuracy",
        "best_val_accuracy": float(validation_metric), "run_id": run_id,
        "created_at": created.isoformat(),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".checkpoint-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print("saved", destination)
    return destination


def load_checkpoint(path, device="cpu"):
    """Restore new .pth and legacy run/state_dict .pt files, without downloading weights."""
    from src.models.factory import build_classifier

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    run = checkpoint["run"]
    kwargs = dict(run.get("encoder_kwargs", {}))
    if run["encoder"] != "scratch":
        kwargs.update(weights=None, freeze=run.get("freeze", False))
    model = build_classifier(run["encoder"], num_classes=checkpoint["state_dict"]["fc.weight"].shape[0],
                             dropout=run["dropout"], **kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), run
