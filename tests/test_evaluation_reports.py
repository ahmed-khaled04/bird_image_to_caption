"""Compatibility of incoming evaluation reports with automatic and legacy checkpoints."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.evaluation import report
from src.models.factory import build_classifier
from src.training import checkpoints


class EvaluationCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / 'models'
        for target in [patch.object(checkpoints, 'MODELS_DIR', self.directory),
                       patch.object(report, 'PROJECT_ROOT', self.root)]:
            target.start()
            self.addCleanup(target.stop)
        self.model = build_classifier('scratch', num_classes=5, widths=(4, 8)).eval()
        self.run = {'name': 'scratch', 'encoder': 'scratch', 'dropout': 0.3,
                    'freeze': False, 'image_size': 32, 'batch_size': 2, 'seed': 42}
        run, labels = checkpoints.training_metadata(
            self.model, {'train': [], 'validation': []}, torch.optim.Adam(self.model.parameters()),
            2, self.run)
        self.path = checkpoints.save_checkpoint(self.model, run, 1, 0.5, labels)

    def test_loads_exact_recorded_checkpoint_with_additional_metadata(self):
        loaded = report.load_model(self.run, self.directory, 'cpu', str(self.path))
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(loaded(image), self.model(image), rtol=0, atol=0)

    def test_relocated_checkpoint_and_missing_reference(self):
        teammate_path = '/teammate/project/models/' + self.path.name
        loaded = report.load_model(self.run, self.directory, 'cpu', teammate_path)
        self.assertFalse(loaded.training)
        with self.assertRaisesRegex(FileNotFoundError, 'copy it'):
            report.load_model(self.run, self.directory, 'cpu', '/teammate/missing.pth')

    def test_old_result_without_checkpoint_path_uses_legacy_location(self):
        legacy_dir = self.root / 'outputs/checkpoints'
        legacy_dir.mkdir(parents=True)
        payload = torch.load(self.path, weights_only=True)
        torch.save(payload, legacy_dir / 'scratch.pt')
        loaded = report.load_model(self.run, self.directory, 'cpu')
        self.assertFalse(loaded.training)

    def test_mismatched_training_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'do not match'):
            report.load_model({**self.run, 'image_size': 64}, self.directory, 'cpu', self.path)

    def test_evaluate_run_forwards_checkpoint_reference(self):
        saved = {'run': self.run, 'checkpoint_path': str(self.path)}
        scores = {'accuracy': 0.5}
        with patch.object(report, 'load_model', return_value=self.model) as load, \
                patch.object(report, 'build_dataloaders', return_value={'validation': []}), \
                patch.object(report, 'evaluate', return_value={}), \
                patch.object(report, 'save_details', return_value=scores):
            actual = report.evaluate_run(saved, 'validation', {'validation': []}, [], self.directory,
                                         self.root / 'tables', self.root / 'figures', 'cpu', 0)
        self.assertEqual(actual, scores)
        load.assert_called_once_with(self.run, self.directory, 'cpu', checkpoint_path=str(self.path))


if __name__ == '__main__':
    unittest.main()
