"""Evidence must distinguish complete tracking from unavailable historical weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.models.factory import build_classifier
from src.training import checkpoint_refs, evidence, tracking


class EvidenceTests(unittest.TestCase):
    def test_export_verifies_events_without_attaching_rerun_weights_to_history(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "results"
            source.mkdir()
            models = root / "models"
            models.mkdir()
            run = {"name": "scratch", "encoder": "scratch", "image_size": 32, "dropout": 0.3}
            model = build_classifier("scratch", num_classes=5, widths=(4, 8)).eval()
            checkpoint = models / "scratch_unique.pth"
            torch.save({"run": {**run, **model.checkpoint_config}, "state_dict": model.state_dict(),
                        "class_labels": list("abcde"), "epoch": 1, "best_val_accuracy": 0.4}, checkpoint)
            row = {"epoch": 1, "train_loss": 1.2, "val_loss": 1.4,
                   "train_accuracy": 0.6, "val_accuracy": 0.4}
            saved = {"run": run, "history": [row], "best_epoch": 1, "best_val_accuracy": 0.4,
                     "minutes": 1, "device": "cpu", "checkpoint_path": str(checkpoint)}
            (source / "local.json").write_text(json.dumps(saved))
            old = {key: value for key, value in saved.items() if key != "checkpoint_path"}
            (source / "historical.json").write_text(json.dumps(old))
            originals = {p.name: p.read_bytes() for p in source.iterdir()}
            with patch.object(checkpoint_refs, "MODELS_DIR", models), \
                    patch.object(checkpoint_refs, "PROJECT_ROOT", root):
                tracking.import_saved_runs([source], root / "logs")
                report = evidence.export_tracking_evidence([source], root / "evidence", root / "logs")
            self.assertEqual(report["checkpoint_status"].tolist(), ["missing", "verified"])
            self.assertTrue(report["curves_verified"].all())
            self.assertTrue(report["config_verified"].all())
            self.assertTrue(report["best_metric_verified"].all())
            self.assertTrue(report["hparams_present"].all())
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, originals)
            inventory = json.loads((root / "evidence/checkpoint_inventory.json").read_text())
            self.assertEqual(len(inventory), 1)
            self.assertEqual(next(iter(inventory.values()))["sha256"], evidence.file_hash(checkpoint))
            for name in ["tracking_accuracy.png", "tracking_loss.png", "tracking_events.json", "coverage.csv"]:
                self.assertTrue((root / "evidence" / name).is_file())

    def test_export_detects_event_curves_that_differ_from_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = {"name": "fixture", "encoder": "scratch"}
            row = {"epoch": 1, "train_loss": 1.2, "val_loss": 1.4,
                   "train_accuracy": 0.6, "val_accuracy": 0.4}
            with tracking.TensorBoardRun(run, log_root=temporary) as record:
                record.log_epoch(row)
                record.complete(1, 0.4, None)
            saved = {"run": run, "history": [{**row, "val_accuracy": 0.5}], "best_val_accuracy": 0.4}
            export = evidence.event_export(record.log_dir, saved)
            self.assertFalse(export["curves_match_source"])
            self.assertTrue(export["config_matches_source"])


if __name__ == "__main__":
    unittest.main()
