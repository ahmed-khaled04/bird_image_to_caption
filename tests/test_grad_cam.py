import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from datasets import Dataset, Features, ClassLabel, Image as HFImage

from src.evaluation.grad_cam import GradCAM
from src.models.factory import build_classifier
from src.grad_cam import generate_report


class ToyEncoder(nn.Module):
    def forward(self, x):
        return [x * 2]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ToyEncoder()

    def forward(self, x):
        features = self.encoder(x)[-1]
        value = features.mean((1, 2, 3))
        return torch.stack([value, -value], dim=1)


class GradCAMTests(unittest.TestCase):
    def test_known_map_and_negative_target(self):
        model = ToyModel().train()
        image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        with torch.no_grad():
            cam, logits, target = GradCAM(model)(image)
        expected = image[0].mean(0)
        expected /= expected.max()
        torch.testing.assert_close(cam, expected)
        self.assertEqual(target, 0)
        self.assertTrue(model.training)
        negative, _, _ = GradCAM(model)(image, 1)
        self.assertEqual(float(negative.max()), 0)
        with self.assertRaises(ValueError):
            GradCAM(model)(image, 2)
        self.assertEqual(len(model.encoder._forward_hooks), 0)
        self.assertTrue(model.training)

    def test_all_project_encoders_including_frozen(self):
        torch.set_num_threads(1)
        for encoder in ["scratch", "resnet18", "resnet50", "efficientnet_b0"]:
            with self.subTest(encoder=encoder):
                kwargs = {} if encoder == "scratch" else {"weights": None, "freeze": True}
                model = build_classifier(encoder, num_classes=3, **kwargs).eval()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                cam, logits, _ = GradCAM(model)(torch.randn(1, 3, 64, 64))
                self.assertEqual(tuple(cam.shape), (64, 64))
                self.assertTrue(torch.isfinite(cam).all())
                self.assertTrue(((cam >= 0) & (cam <= 1)).all())
                self.assertFalse(cam.requires_grad)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name])

    def test_report_checkpoint_and_both_outcomes(self):
        torch.set_num_threads(1)
        model = build_classifier("scratch", num_classes=2).eval()
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.copy_(torch.tensor([1., 0.]))
        split = Dataset.from_dict({"image": [Image.fromarray(np.full((40, 50, 3), 128, dtype=np.uint8))] * 2,
                                   "label": [0, 1]},
                                  features=Features({"image": HFImage(), "label": ClassLabel(names=["a", "b"])}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scratch.pt"
            torch.save({"run": {"name": "scratch", "encoder": "scratch", "dropout": 0.3,
                                "image_size": 32}, "state_dict": model.state_dict()}, path)
            report = generate_report(path, split, directory, examples=1)
            self.assertEqual(report["selected"], {"correct": 1, "wrong": 1})
            self.assertEqual(report["scanned"], 2)
            for example in report["examples"]:
                with Image.open(Path(directory) / "scratch" / example["figure"]) as figure:
                    self.assertGreater(figure.width, 100)
                self.assertEqual(example["target_class"], 0)
            stored = json.loads((Path(directory) / "scratch/report.json").read_text())
            self.assertEqual(stored, report)
            limited = generate_report(path, split, directory, examples=2, max_images=1)
            self.assertEqual(limited["selected"], {"correct": 1, "wrong": 0})


if __name__ == "__main__":
    unittest.main()
