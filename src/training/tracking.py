"""TensorBoard logging shared by scratch, transfer, tuning, and notebook runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from src.training.checkpoints import PROJECT_ROOT


LOGS_DIR = PROJECT_ROOT / "outputs" / "tensorboard"


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:80] or "run"


def _hparams(config, prefix=""):
    """HParams accepts scalar values; retain nested/list/None settings explicitly."""
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            result.update(_hparams(value, name))
        else:
            result[name] = value if isinstance(value, (str, bool, int, float)) else json.dumps(value)
    return result


def _summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError("TensorBoard logging requires tensorboard. Run: python -m pip install -r requirements.txt") from error
    return SummaryWriter


class TensorBoardRun:
    """One unique event directory per run, with curves flushed after every epoch."""

    def __init__(self, run, class_labels=None, log_root=None, *, source="training", run_name=None):
        root = Path(log_root) if log_root is not None else LOGS_DIR
        name = run_name or f"{_slug(run['encoder'])}_{_slug(run['name'])}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}"
        self.log_dir = root.resolve() / name
        writer_class = _summary_writer()
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.writer = writer_class(log_dir=str(self.log_dir), flush_secs=10)
        self.run = run
        self.started = time.perf_counter()
        self.metrics = {}
        self.record = {"run": run, "source": source, "status": "running", "completed_epochs": 0,
                       "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            self.writer.add_text("run/config", "```json\n" + json.dumps(run, indent=2) + "\n```")
            self.writer.add_text("run/source", source)
            if class_labels is not None:
                self.writer.add_text("run/class_labels", json.dumps(class_labels))
            self._save_record()
            self.writer.flush()
            print("TensorBoard:", self.log_dir)
        except BaseException:
            self.writer.close()
            raise

    def _save_record(self):
        (self.log_dir / "run.json").write_text(json.dumps(self.record, indent=2))

    def __enter__(self):
        return self

    def log_epoch(self, row, *, learning_rates=(), validation_top5=None, seconds=None):
        epoch = int(row["epoch"])
        for key, tag in [("train_loss", "loss/train"), ("val_loss", "loss/validation"),
                         ("train_accuracy", "accuracy/train"), ("val_accuracy", "accuracy/validation")]:
            self.writer.add_scalar(tag, row[key], epoch)
        if validation_top5 is not None:
            self.writer.add_scalar("accuracy/validation_top5", validation_top5, epoch)
        if seconds is not None:
            self.writer.add_scalar("time/epoch_seconds", seconds, epoch)
        for group, rate in enumerate(learning_rates):
            self.writer.add_scalar(f"learning_rate/group_{group}", rate, epoch)
        self.record["completed_epochs"] = epoch
        self.writer.flush()

    def complete(self, best_epoch, best_val_accuracy, checkpoint_path, *, seconds=None):
        self.metrics = {"hparams/best_val_accuracy": float(best_val_accuracy), "hparams/best_epoch": int(best_epoch)}
        self.writer.add_scalar("summary/best_val_accuracy", best_val_accuracy, best_epoch)
        self.writer.add_scalar("summary/best_epoch", best_epoch, best_epoch)
        self.writer.add_text("run/checkpoint", str(checkpoint_path) if checkpoint_path else "Not recorded in source results")
        self.writer.add_hparams(_hparams(self.run), self.metrics, run_name=".")
        if seconds is not None:
            self.writer.add_scalar("time/training_seconds", seconds, best_epoch)
        self.record.update(status="completed", best_epoch=int(best_epoch),
                           best_val_accuracy=float(best_val_accuracy),
                           checkpoint_path=str(checkpoint_path) if checkpoint_path else None)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None:
                self.record["status"] = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
                self.writer.add_text("run/error", f"{exc_type.__name__}: {exc}")
            self.record["ended_at"] = datetime.now(timezone.utc).isoformat()
            self.record["elapsed_seconds"] = time.perf_counter() - self.started
            self.writer.add_text("run/status", self.record["status"])
            self._save_record()
            self.writer.flush()
        finally:
            self.writer.close()


def log_test_metrics(log_dir, metrics, epoch):
    """Append actual held-out metrics to the same run after best-state restoration."""
    with _summary_writer()(log_dir=str(log_dir)) as writer:
        for key, value in metrics.items():
            writer.add_scalar(f"test/{key.removeprefix('test_')}", value, epoch)


def import_saved_runs(runs_dirs, log_root=None):
    """Export existing real training histories without training or creating checkpoints.

    Content-based names make repeated imports of the same results idempotent. The
    source JSON and its recorded checkpoint references are preserved verbatim.
    """
    from hashlib import sha256

    root = (Path(log_root) if log_root is not None else LOGS_DIR) / "imported"
    paths = sorted({path.resolve() for directory in runs_dirs for path in Path(directory).glob("*.json")})
    if not paths:
        raise ValueError("No saved run JSON files found in the supplied directories")
    imported = []
    for path in paths:
        content = path.read_bytes()
        saved = json.loads(content)
        if saved.get("device") == "synthetic":
            print("Skipping synthetic result:", path)
            continue
        if not saved.get("history") or "run" not in saved:
            raise ValueError(f"{path} must contain a run configuration and non-empty epoch history")
        run = saved["run"]
        name = f"{_slug(run['encoder'])}_{_slug(run['name'])}_{sha256(content).hexdigest()}"
        destination = root.resolve() / name
        if destination.exists():
            record_file = destination / "run.json"
            if record_file.is_file() and json.loads(record_file.read_text()).get("status") == "completed":
                print("Already imported:", path)
                imported.append(destination)
                continue
            raise FileExistsError(f"Incomplete import at {destination}; use a different --log-dir to retry")
        checkpoint = saved.get("checkpoint_path")
        if not checkpoint:
            legacy = PROJECT_ROOT / "outputs" / "checkpoints" / f"{run['name']}.pt"
            checkpoint = str(legacy) if legacy.is_file() else None
        with TensorBoardRun(run, log_root=root, source=f"historical:{path}", run_name=name) as tracking:
            for row in saved["history"]:
                tracking.log_epoch(row)
            tracking.complete(saved["best_epoch"], saved["best_val_accuracy"], checkpoint,
                              seconds=saved["minutes"] * 60 if "minutes" in saved else None)
        metrics = {key: value for key, value in saved.items()
                   if key.startswith("test_") and isinstance(value, (int, float))}
        if metrics:
            log_test_metrics(tracking.log_dir, metrics, saved["best_epoch"])
        imported.append(tracking.log_dir)
    return imported
