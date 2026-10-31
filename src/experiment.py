"""One training run from a config dict. Transfer learning (point 3) and tuning (point 4) both go through
train_run, so they train, time and score every model the same way.

The epoch loop and the per-split evaluation are the training code of points 1-2, imported in one place
(training_code). Everything around them is here: seeding, building the model and optimizer, timing,
loading the best epoch, test metrics and checkpoints.

A run is a dict with: name, encoder, freeze, lr, weight_decay, dropout, optimizer, epochs, batch_size,
image_size, seed and num_workers.
"""
import random
import time
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from src.data.loaders import build_dataloaders
from src.data.loader import class_names
from src.models.factory import build_classifier

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,  # weight decay applied to the weights directly, not mixed into Adam's gradient scaling
    "sgd": lambda params, lr, weight_decay: torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9),
}


def training_code():
    """The epoch loop and evaluation from points 1-2:

    fit(model, loaders, loss_fn, optimizer, device, epochs, **metadata) -> {"history": [one dict per epoch with epoch,
        train_loss, train_accuracy, val_loss, val_accuracy], "best_val_accuracy", "best_epoch", "best_state", "checkpoint_path"}
    evaluate(model, loader, loss_fn, device) -> {"loss", "accuracy", "top5_accuracy", "labels", "predictions"}
    """
    try:
        from src.evaluation.classification import evaluate
        from src.training.trainer import fit
    except ImportError as error:
        raise ImportError("the training code is not in src yet: needs fit in src/training/trainer.py and "
                          f"evaluate in src/evaluation/classification.py ({error})") from error
    return fit, evaluate


def load_processed(data_dir: Path, limit: int = None) -> DatasetDict:
    """The cleaned splits saved by `python -m src.cli clean`. `limit` keeps the first images of each split,
    for a quick check that everything runs."""
    data = load_from_disk(str(data_dir))
    if limit:
        data = DatasetDict({name: split.select(range(min(limit, len(split)))) for name, split in data.items()})
    print("data:", ", ".join(f"{name} {len(split)}" for name, split in data.items()))
    return data


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(run: dict) -> nn.Module:
    # the scratch encoder has no pretrained weights, so it takes no freeze option
    encoder_kwargs = {} if run["encoder"] == "scratch" else {"freeze": run["freeze"]}
    return build_classifier(run["encoder"], dropout=run["dropout"], **encoder_kwargs)


def build_optimizer(model: nn.Module, run: dict) -> torch.optim.Optimizer:
    """Only the trainable weights: with a frozen encoder that is just the classification layer."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return OPTIMIZERS[run["optimizer"]](trainable, lr=run["lr"], weight_decay=run["weight_decay"])


def count_parameters(model: nn.Module) -> tuple:
    """(trainable, total) number of weights."""
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


def test_metrics(result: dict) -> dict:
    labels, predictions = result["labels"], result["predictions"]
    return {
        "test_accuracy": float(result["accuracy"]),
        "test_top5_accuracy": float(result["top5_accuracy"]),
        # macro: every species counts the same, however many test images it has (11 to 30)
        "test_precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def mode_name(run: dict) -> str:
    if run["encoder"] == "scratch":
        return "from scratch"
    return "frozen" if run["freeze"] else "fine-tune"


def train_run(run: dict, data: DatasetDict, test: bool = False, checkpoint_dir: Path = None) -> dict:
    """Train on train, keep the epoch with the best validation accuracy, and score it once on test if `test`.
    Tuning trials pass test=False: settings are chosen on validation only."""
    fit, evaluate = training_code()
    device = pick_device()
    random.seed(run["seed"])
    torch.manual_seed(run["seed"])  # same starting head for every run with this seed

    model = build_model(run).to(device)
    optimizer = build_optimizer(model, run)
    loss_fn = nn.CrossEntropyLoss()
    splits = ["train", "validation"] + (["test"] if test else [])
    loaders = build_dataloaders(DatasetDict({s: data[s] for s in splits}), image_size=run["image_size"],
                                batch_size=run["batch_size"], num_workers=run["num_workers"], seed=run["seed"])
    trainable, total = count_parameters(model)
    print(f"\n=== {run['name']}: {run['encoder']} {mode_name(run)}, {trainable:,} of {total:,} weights trainable, "
          f"{run['epochs']} epochs on {device}")

    start = time.perf_counter()
    trained = fit(model, loaders, loss_fn, optimizer, device, run["epochs"], run_config=run,
                  class_labels=class_names(data["train"]), checkpoint_dir=checkpoint_dir)
    seconds = time.perf_counter() - start
    model.load_state_dict(trained["best_state"])  # back to the best epoch, not the last one

    result = {
        "run": run, "mode": mode_name(run), "trainable_params": trainable, "total_params": total,
        "best_val_accuracy": float(trained["best_val_accuracy"]), "best_epoch": int(trained["best_epoch"]),
        "minutes": seconds / 60, "device": str(device), "history": trained["history"],
        "checkpoint_path": trained["checkpoint_path"],
    }
    if test:
        result.update(test_metrics(evaluate(model, loaders["test"], loss_fn, device)))
    return result


def synthetic_run(run: dict, test: bool = False, checkpoint_dir: Path = None) -> dict:
    """Stand-in for train_run with made-up learning curves, to test the reports in seconds without data or a GPU."""
    rng = random.Random(run["name"])
    final = rng.uniform(0.2, 0.8)
    history = []
    for epoch in range(1, run["epochs"] + 1):
        val = final * (1 - 0.7 ** epoch) + rng.uniform(-0.02, 0.02)
        history.append({"epoch": epoch, "train_loss": 5.3 * 0.75 ** epoch, "train_accuracy": min(1.0, val * 1.2),
                        "val_loss": 5.3 * 0.8 ** epoch + 0.5, "val_accuracy": val})
    best = max(history, key=lambda h: h["val_accuracy"])
    # the weight counts are real: the model is built, only without downloading the pretrained weights
    kwargs = {} if run["encoder"] == "scratch" else {"freeze": run["freeze"], "weights": None}
    trainable, total = count_parameters(build_classifier(run["encoder"], dropout=run["dropout"], **kwargs))
    result = {
        "run": run, "mode": mode_name(run), "trainable_params": trainable, "total_params": total,
        "best_val_accuracy": best["val_accuracy"], "best_epoch": best["epoch"],
        "minutes": rng.uniform(1, 30), "device": "synthetic", "history": history,
    }
    if test:
        result.update({"test_accuracy": final - 0.03, "test_top5_accuracy": min(1.0, final + 0.15),
                       "test_precision_macro": final - 0.02, "test_recall_macro": final - 0.03,
                       "test_f1_macro": final - 0.035})
    return result
