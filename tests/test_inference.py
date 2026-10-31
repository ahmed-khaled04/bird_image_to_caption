"""Checkpoint identity and real image inference regression tests; no downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from src.data.transforms import eval_transform
from src.evaluation.report import load_model
from src.models.factory import build_classifier
from src.predict import predict_image
from src.training import checkpoint_refs


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.models = self.root / "models"
        self.models.mkdir()
        for target in [patch.object(checkpoint_refs, "MODELS_DIR", self.models),
                       patch.object(checkpoint_refs, "PROJECT_ROOT", self.root)]:
            target.start()
            self.addCleanup(target.stop)
        torch.manual_seed(42)
        self.model = build_classifier("scratch", num_classes=5, widths=(4, 8)).eval()
        self.run = {"name": "trial", "encoder": "scratch", "dropout": 0.3, "image_size": 32}
        self.names = ["albatross", "auklet", "blackbird", "catbird", "grebe"]
        self.payload = {"run": {**self.run, **self.model.checkpoint_config},
                        "state_dict": self.model.state_dict(), "class_labels": self.names,
                        "epoch": 2, "best_val_accuracy": 0.8}
        self.path = self.models / "scratch_trial_unique_best.pth"
        torch.save(self.payload, self.path)
        self.saved = {"run": self.run, "best_epoch": 2, "best_val_accuracy": 0.8,
                      "checkpoint_path": str(self.path)}
        self.image_path = self.root / "bird.png"
        Image.new("L", (71, 49), color=123).save(self.image_path)

    def test_enriched_checkpoint_restores_identical_logits_and_label_order(self):
        model = load_model(self.run, self.models, "cpu", self.saved, names=self.names)
        inputs = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), self.model(inputs), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "index order"):
            load_model(self.run, self.models, "cpu", self.saved, names=list(reversed(self.names)))

    def test_wrong_settings_best_epoch_and_score_are_rejected(self):
        for update, expected in [({"run": {**self.run, "dropout": 0.5}}, "dropout"),
                                 ({"best_epoch": 1}, "best epoch"),
                                 ({"best_val_accuracy": 0.7}, "validation accuracy")]:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, expected):
                checkpoint_refs.resolve_checkpoint({**self.saved, **update})

    def test_relocated_reference_and_legacy_checkpoint(self):
        moved = {**self.saved, "checkpoint_path": "/another/computer/" + self.path.name}
        self.assertEqual(checkpoint_refs.resolve_checkpoint(moved), self.path)
        relative = {**self.saved, "checkpoint_path": "models/" + self.path.name}
        self.assertEqual(checkpoint_refs.resolve_checkpoint(relative), self.path)
        legacy = self.models / "trial.pt"
        torch.save({"run": self.run, "state_dict": self.model.state_dict()}, legacy)
        self.assertEqual(checkpoint_refs.resolve_checkpoint({"run": self.run}), legacy)

    def test_missing_historical_reference_never_claims_same_name_rerun(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_refs.resolve_checkpoint({"run": self.run})
        torch.save(self.payload, self.models / "trial.pt")
        with self.assertRaises(FileNotFoundError):
            checkpoint_refs.resolve_checkpoint({**self.saved, "checkpoint_path": "models/missing.pth"})

    def test_ambiguous_legacy_files_are_rejected(self):
        for suffix in [".pt", ".pth"]:
            torch.save(self.payload, self.models / ("trial" + suffix))
        with self.assertRaisesRegex(ValueError, "Multiple checkpoints"):
            checkpoint_refs.resolve_checkpoint({"run": self.run})

    def test_prediction_matches_manual_evaluation_transform_and_softmax(self):
        with Image.open(self.image_path) as image, torch.inference_mode():
            expected = self.model(eval_transform(32)(image).unsqueeze(0)).softmax(1)[0]
        result = predict_image(self.path, self.image_path, top_k=10)
        self.assertEqual(len(result["predictions"]), 5)
        probabilities = [row["probability"] for row in result["predictions"]]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertAlmostEqual(sum(probabilities), 1, places=6)
        for row in result["predictions"]:
            self.assertEqual(row["species"], self.names[row["class_id"]])
            self.assertAlmostEqual(row["probability"], expected[row["class_id"]].item(), places=7)
        with self.assertRaisesRegex(ValueError, "positive"):
            predict_image(self.path, self.image_path, top_k=0)

    def test_legacy_prediction_requires_explicit_labels(self):
        payload = {k: v for k, v in self.payload.items() if k != "class_labels"}
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, "lacks class labels"):
            predict_image(self.path, self.image_path)
        labels = self.root / "labels.json"
        labels.write_text(json.dumps(self.names))
        self.assertEqual(len(predict_image(self.path, self.image_path, labels_path=labels)["predictions"]), 5)


if __name__ == "__main__":
    unittest.main()
