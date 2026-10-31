"""Shared training loop for scratch and pretrained classifiers."""
import copy
import time

from src.evaluation.classification import evaluate
from src.training.checkpoints import checkpoint_directory, save_checkpoint, training_metadata
from src.training.tracking import TensorBoardRun


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        seen += batch_size

    if seen == 0:
        raise ValueError("Training loader is empty")

    return total_loss / seen, correct / seen


def fit(model, loaders, loss_fn, optimizer, device, epochs, *, run_config=None, class_labels=None,
        checkpoint_dir=None, tensorboard_dir=None):
    """Train unchanged; automatically save the best validation state of every completed run."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    directory = checkpoint_directory(checkpoint_dir)
    run, labels = training_metadata(model, loaders, optimizer, epochs, run_config, class_labels)
    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = None

    training_start = time.perf_counter()
    with TensorBoardRun(run, labels, tensorboard_dir) as tracking:
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()

            train_loss, train_accuracy = train_one_epoch(
                model, loaders["train"], loss_fn, optimizer, device
            )
            val = evaluate(
                model, loaders["validation"], loss_fn, device
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
            })

            tracking.log_epoch(history[-1], learning_rates=[group["lr"] for group in optimizer.param_groups],
                               validation_top5=val.get("top5_accuracy"), seconds=time.perf_counter() - start)

            if val["accuracy"] > best_val_accuracy:
                best_val_accuracy = val["accuracy"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"epoch {epoch}/{epochs} | "
                f"train loss {train_loss:.3f} acc {train_accuracy:.3f} | "
                f"val loss {val['loss']:.3f} acc {val['accuracy']:.3f} | "
                f"{time.perf_counter() - start:.0f}s"
            )

        if best_state is None:
            raise ValueError("No valid best validation state was produced")
        checkpoint_path = save_checkpoint(model, run, best_epoch, best_val_accuracy, labels,
                                          state_dict=best_state, directory=directory)
        tracking.complete(best_epoch, best_val_accuracy, checkpoint_path, seconds=time.perf_counter() - training_start)
    return {
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "checkpoint_path": str(checkpoint_path),
        "tensorboard_dir": str(tracking.log_dir),
    }
