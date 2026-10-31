"""Export report evidence from actual TensorBoard events and audit checkpoint coverage."""
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData

from src.training.checkpoint_refs import read_checkpoint, resolve_checkpoint
from src.training.checkpoints import PROJECT_ROOT, load_checkpoint
from src.training.tracking import LOGS_DIR, _slug


CURVES = {"loss/train": "train_loss", "loss/validation": "val_loss",
          "accuracy/train": "train_accuracy", "accuracy/validation": "val_accuracy"}


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_export(directory, saved):
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0, "tensors": 0}).Reload()
    tags = events.Tags()
    curves = {tag: [{"epoch": item.step, "value": item.value} for item in events.Scalars(tag)]
              for tag in CURVES if tag in tags["scalars"]}
    complete = all(tag in curves and len(curves[tag]) == len(saved["history"]) and all(
        point["epoch"] == row["epoch"] and math.isclose(point["value"], row[key], rel_tol=1e-6, abs_tol=1e-6)
        for point, row in zip(curves[tag], saved["history"])) for tag, key in CURVES.items())
    text = {tag.removesuffix("/text_summary"): events.Tensors(tag)[-1].tensor_proto.string_val[0].decode()
            for tag in tags["tensors"] if tag.endswith("/text_summary")}
    try:
        plugin = events.PluginTagToContent("hparams")
    except KeyError:
        plugin = {}
    hparams = []
    for content in plugin.values():
        data = HParamsPluginData.FromString(content)
        if data.HasField("session_start_info"):
            hparams.append({key: getattr(value, value.WhichOneof("kind"))
                            for key, value in data.session_start_info.hparams.items()})
    best = events.Scalars("summary/best_val_accuracy") if "summary/best_val_accuracy" in tags["scalars"] else []
    metrics_match = bool(best) and math.isclose(best[-1].value, saved["best_val_accuracy"], rel_tol=1e-6, abs_tol=1e-6)
    config = text.get("run/config", "null")
    if config.startswith("```json\n") and config.endswith("\n```"):
        config = config[len("```json\n"):-len("\n```")]
    try:
        logged_config = json.loads(config)
        config_match = isinstance(logged_config, dict) and all(
            key in logged_config and logged_config[key] == value for key, value in saved["run"].items())
    except json.JSONDecodeError:
        config_match = False
    return {"curves": curves, "text": text, "hparams": hparams,
            "curves_match_source": complete, "config_matches_source": config_match,
            "best_metric_matches_source": metrics_match,
            "scalar_tags": tags["scalars"], "log_directory": display_path(directory)}


def export_tracking_evidence(runs_dirs, output_dir, log_root=None, checkpoint_dir=None):
    """Keep original result files untouched; explicitly report unavailable historical weights."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(log_root) if log_root else LOGS_DIR
    paths = sorted({path.resolve() for directory in runs_dirs for path in Path(directory).glob("*.json")})
    if not paths:
        raise ValueError("No saved run JSON files found")
    rows, settings, exports, inventory = [], [], {}, {}
    for path in paths:
        saved = json.loads(path.read_text())
        if saved.get("device") == "synthetic":
            continue
        if not saved.get("history"):
            raise ValueError(f"Missing real epoch history in {path}")
        identifier = display_path(path)
        source_hash = file_hash(path)
        run = saved["run"]
        row = {"source": identifier, "source_sha256": source_hash, "run": run["name"],
               "epochs": len(saved["history"]), "best_epoch": saved["best_epoch"],
               "best_val_accuracy": saved["best_val_accuracy"], "checkpoint_status": "missing",
               "checkpoint_path": "", "checkpoint_sha256": "", "checkpoint_issue": ""}
        try:
            checkpoint = resolve_checkpoint(saved, checkpoint_dir, path)
            # Strict state-dict reconstruction verifies architecture compatibility too.
            model, _ = load_checkpoint(checkpoint, "cpu")
            payload = read_checkpoint(checkpoint)
            labels = payload.get("class_labels")
            if labels is not None and len(labels) != model.fc.out_features:
                raise ValueError("Checkpoint label count does not match classifier outputs")
            del model
            row.update(checkpoint_status="verified", checkpoint_path=display_path(checkpoint),
                       checkpoint_sha256=file_hash(checkpoint))
            item = inventory.setdefault(row["checkpoint_path"], {
                "sha256": row["checkpoint_sha256"], "bytes": checkpoint.stat().st_size,
                "run": payload["run"], "epoch": payload.get("epoch"),
                "best_val_accuracy": payload.get("best_val_accuracy"),
                "class_labels": labels, "verified_sources": [],
            })
            item["verified_sources"].append(identifier)
        except FileNotFoundError as error:
            row["checkpoint_issue"] = str(error)
        except (ValueError, RuntimeError, KeyError) as error:
            row.update(checkpoint_status="invalid", checkpoint_issue=str(error))
        # Match historical imports by exact source content, not shared run names.
        imported = root / "imported" / f"{_slug(run['encoder'])}_{_slug(run['name'])}_{source_hash}"
        directory = imported
        if not directory.is_dir() and saved.get("tensorboard_dir"):
            directory = Path(saved["tensorboard_dir"])
        if directory.is_dir():
            export = event_export(directory, saved)
            exports[identifier] = {"source_sha256": source_hash, **export}
            row.update(curves_verified=export["curves_match_source"],
                       config_verified=export["config_matches_source"],
                       best_metric_verified=export["best_metric_matches_source"],
                       hparams_present=bool(export["hparams"]),
                       tracking_status=export["text"].get("run/status", "unknown"))
        else:
            row.update(curves_verified=False, config_verified=False, best_metric_verified=False,
                       hparams_present=False, tracking_status="missing")
            exports[identifier] = {"source_sha256": source_hash, "error": "No matching TensorBoard logs"}
        rows.append(row)
        settings.append({"source": identifier, **run})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(settings).to_csv(output / "run_settings.csv", index=False)
    (output / "tracking_events.json").write_text(json.dumps(exports, indent=2) + "\n")
    (output / "checkpoint_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    for kind in ["loss", "accuracy"]:
        count = len(exports)
        if not count:
            break
        fig, axes = plt.subplots(math.ceil(count / 3), min(3, count),
                                 figsize=(15, 3.6 * math.ceil(count / 3)), squeeze=False,
                                 constrained_layout=True)
        for axis, (identifier, export) in zip(axes.flat, exports.items()):
            for split, colour in [("train", "#2864b4"), ("validation", "#d25b24")]:
                points = export.get("curves", {}).get(f"{kind}/{split}", [])
                if points:
                    axis.plot([point["epoch"] for point in points], [point["value"] for point in points],
                              label=split, color=colour)
            location = "local" if "/my_training/" in identifier else "team"
            axis.set_title(f"{location}: {Path(identifier).stem}", fontsize=10)
            axis.set(xlabel="Epoch", ylabel="Accuracy (fraction)" if kind == "accuracy" else "Cross-entropy loss")
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=8)
        for axis in list(axes.flat)[count:]:
            axis.set_visible(False)
        fig.suptitle(f"{kind.title()} exported from TensorBoard events (historical imports)")
        fig.savefig(output / f"tracking_{kind}.png", dpi=150)
        plt.close(fig)
    verified = sum(row["checkpoint_status"] == "verified" for row in rows)
    lines = ["# Experiment tracking evidence", "",
             "Generated from actual TensorBoard event files; curves are checked against the original saved histories.", "",
             f"- Real result records audited: **{len(rows)}**.",
             f"- Checkpoints verified against their referenced result records: **{verified}**.",
             f"- Records with unavailable/invalid original weights: **{len(rows) - verified}**.",
             "- Source JSON files are preserved; SHA-256 values identify the exact inputs and weights.",
             "- Matching names, settings, or scores do not establish that a rerun is an original run.", "",
             "| Source | Curves | Configuration | Best metric | HParams | Weights |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        check = lambda key: "verified" if row[key] else "missing/mismatch"
        lines.append(f"| `{row['source']}` | {check('curves_verified')} | {check('config_verified')} | "
                     f"{check('best_metric_verified')} | {'present' if row['hparams_present'] else 'missing'} | {row['checkpoint_status']} |")
    lines += ["", "![Training and validation accuracy](tracking_accuracy.png)", "",
              "![Training and validation loss](tracking_loss.png)", "", "## Files", "",
              "- `coverage.csv`: run-by-run checks, missing-weight reasons, and hashes.",
              "- `run_settings.csv`: original hyperparameters for comparison.",
              "- `tracking_events.json`: exported curves, text, and HParams read from TensorBoard.",
              "- `checkpoint_inventory.json`: verified checkpoint metadata, labels, size, and SHA-256.", "",
              "## Recovery procedure", "",
              "For each missing historical run, obtain its original checkpoint from the teammate's training environment "
              "or model storage. Put it in `models/` and add its exact `checkpoint_path` to that run's JSON. "
              "Keep a provenance note from the owner; do not attach a new rerun's weights to an old history. "
              "Reimport tracking and rerun this export. Metadata checks reject setting/epoch/score mismatches; "
              "matching metadata alone is not proof of historical provenance. New training is recorded separately.", "",
              "## Scope", "",
              "These are historical imports, not live tracking captured during the original training. "
              "They expose only values present in the source histories; missing old checkpoints, learning-rate "
              "schedules, or unrecorded metrics are not reconstructed. Checkpoints are Git-ignored and must be "
              "shared separately. Evidence exports do not include the weight tensors."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(coverage.to_string(index=False))
    print("Evidence exported to", output)
    return coverage
