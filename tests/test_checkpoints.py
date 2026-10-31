"""Checkpoint coverage with tiny fixtures and mocked epoch work; no bird model retraining."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from datasets import ClassLabel, Dataset, DatasetDict, Features

from src import experiment
from src.cli import output_dirs
from src.data.transforms import eval_transform
from src.evaluation.classification import evaluate
from src.evaluation.grad_cam import GradCAM
from src.grad_cam import load_checkpoint as gradcam_load_checkpoint
from src.models.factory import build_classifier
from src.training import checkpoints, tracking
from src.training.search import compare_searches, run_search
from src.training.trainer import fit
from src.transfer import comparison_table, run_and_save
from src.tuning import make_objective, retrain_best


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / 'models'
        directory_patch = patch.object(checkpoints, 'MODELS_DIR', self.directory)
        directory_patch.start()
        self.addCleanup(directory_patch.stop)
        tracking_patch = patch.object(tracking, 'LOGS_DIR', self.root / 'tensorboard')
        tracking_patch.start()
        self.addCleanup(tracking_patch.stop)
        self.labels = ['a', 'b', 'c', 'd', 'e']
        split = Dataset.from_dict({'label': [0, 1, 2, 3, 4]},
                                  features=Features({'label': ClassLabel(names=self.labels)}))
        self.data = DatasetDict({name: split for name in ['train', 'validation', 'test']})
        dataset = SimpleNamespace(split=split, transform=eval_transform(32))
        self.loaders = {name: SimpleNamespace(dataset=dataset, batch_size=2, num_workers=0)
                        for name in self.data}

    def tiny_model(self, **kwargs):
        return build_classifier('scratch', num_classes=5, widths=(4, 8), **kwargs)

    def run_config(self, encoder='scratch', freeze=False):
        return {'name': f'{encoder}_experiment', 'encoder': encoder, 'freeze': freeze,
                'dropout': 0.3, 'image_size': 32, 'epochs': 2, 'seed': 42,
                'batch_size': 2, 'num_workers': 0, 'optimizer': 'adam',
                'lr': 0.001, 'weight_decay': 0.0}

    def metadata(self, model, run=None):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return checkpoints.training_metadata(model, self.loaders, optimizer, 2, run)

    def test_scratch_and_transfer_exact_round_trip_and_gradcam(self):
        for encoder, freeze in [('scratch', False), ('resnet18', True), ('resnet18', False)]:
            with self.subTest(encoder=encoder, freeze=freeze):
                model = (self.tiny_model() if encoder == 'scratch' else
                         build_classifier(encoder, num_classes=5, weights=None, freeze=freeze)).eval()
                original = self.run_config(encoder, freeze)
                run, labels = self.metadata(model, original)
                path = checkpoints.save_checkpoint(model, run, 2, np.float64(0.8), labels)
                payload = torch.load(path, weights_only=True)
                self.assertEqual(payload['epoch'], 2)
                self.assertEqual(payload['validation_metric'], 0.8)
                self.assertEqual(payload['class_labels'], self.labels)
                self.assertEqual(payload['run']['num_classes'], 5)
                self.assertEqual(path.suffix, '.pth')
                self.assertEqual(path.parent, self.directory)
                self.assertNotIn('num_classes', original)
                restored, loaded_run = gradcam_load_checkpoint(path, 'cpu')
                images = torch.randn(2, 3, 32, 32)
                with torch.no_grad():
                    torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)
                self.assertEqual(loaded_run['freeze'], freeze)
                cam, _, _ = GradCAM(restored)(images[:1])
                self.assertTrue(torch.isfinite(cam).all())
                result = evaluate(restored, [(images, torch.tensor([0, 1]))], nn.CrossEntropyLoss(), 'cpu')
                self.assertEqual(len(result['predictions']), 2)

    def test_unique_names_and_failures_never_replace_existing_files(self):
        model = self.tiny_model()
        run, labels = self.metadata(model)
        first = checkpoints.save_checkpoint(model, run, 1, 0.5, labels)
        original = first.read_bytes()
        second = checkpoints.save_checkpoint(model, run, 1, 0.5, labels)
        self.assertNotEqual(first, second)
        legacy = self.directory / 'old.pt'
        legacy.write_bytes(b'legacy checkpoint untouched')

        def fail_during_write(payload, file):
            file.write(b'partial')
            raise OSError('disk write failed')

        with patch.object(checkpoints.torch, 'save', side_effect=fail_during_write):
            with self.assertRaisesRegex(OSError, 'disk write failed'):
                checkpoints.save_checkpoint(model, run, 1, 0.5, labels)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(legacy.read_bytes(), b'legacy checkpoint untouched')
        self.assertEqual(len(list(self.directory.glob('*.pth'))), 2)
        self.assertEqual(list(self.directory.glob('*.tmp')), [])

    def test_atomic_publication_refuses_a_filename_collision(self):
        model = self.tiny_model()
        run, labels = self.metadata(model)
        fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch.object(checkpoints, 'datetime') as clock, patch.object(checkpoints, 'uuid4') as uuid:
            clock.now.return_value = fixed_time
            uuid.return_value.hex = 'same-id'
            path = checkpoints.save_checkpoint(model, run, 1, 0.4, labels)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                checkpoints.save_checkpoint(model, run, 2, 0.9, labels)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(self.directory.glob('*.tmp')), [])

    def test_fit_automatically_saves_best_epoch_without_manual_metadata(self):
        model = self.tiny_model()
        optimizer = torch.optim.Adam(model.parameters())
        epoch = 0

        def fake_epoch(*args):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                model.fc.bias.fill_(epoch)
            return 1.0, 0.25

        scores = [{'loss': 1.0, 'accuracy': value} for value in [0.3, 0.8, 0.6]]
        with patch('src.training.trainer.train_one_epoch', side_effect=fake_epoch), \
                patch('src.training.trainer.evaluate', side_effect=scores):
            result = fit(model, self.loaders, nn.CrossEntropyLoss(), optimizer, 'cpu', 3)
        path = Path(result['checkpoint_path'])
        payload = torch.load(path, weights_only=True)
        self.assertEqual(payload['epoch'], 2)
        self.assertEqual(payload['validation_metric'], 0.8)
        self.assertEqual(payload['run']['image_size'], 32)
        self.assertEqual(payload['class_labels'], self.labels)
        torch.testing.assert_close(payload['state_dict']['fc.bias'], torch.full((5,), 2.))
        torch.testing.assert_close(model.fc.bias, torch.full((5,), 3.))
        self.assertEqual(len(list(self.directory.glob('*.pth'))), 1)

    def test_interrupted_fit_does_not_publish_or_overwrite_weights(self):
        model = self.tiny_model()
        run, labels = self.metadata(model)
        first = checkpoints.save_checkpoint(model, run, 1, 0.6, labels)
        original = first.read_bytes()
        with patch('src.training.trainer.train_one_epoch', side_effect=[(1., 0.2), KeyboardInterrupt]), \
                patch('src.training.trainer.evaluate', return_value={'loss': 1., 'accuracy': 0.5}):
            with self.assertRaises(KeyboardInterrupt):
                fit(model, self.loaders, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), 'cpu', 2)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(list(self.directory.glob('*.pth')), [first])

    def test_legacy_pt_loads_without_new_metadata(self):
        self.directory.mkdir()
        # An old standard scratch model, not the custom-width fixture, has no encoder_kwargs.
        model = build_classifier('scratch', num_classes=5).eval()
        run = {'name': 'legacy', 'encoder': 'scratch', 'dropout': 0.3, 'image_size': 32}
        path = self.directory / 'legacy.pt'
        torch.save({'state_dict': model.state_dict(), 'run': run, 'best_val_accuracy': 0.4}, path)
        restored, loaded_run = gradcam_load_checkpoint(path, 'cpu')
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)
        self.assertEqual(loaded_run, run)

    def test_metadata_handles_search_scalars_without_unsafe_pickle_globals(self):
        run, labels = self.metadata(self.tiny_model(), {'name': 'trial', 'lr': np.float64(0.001),
                                                     'seed': np.int64(42), 'config_file': Path('config.yaml')})
        path = checkpoints.save_checkpoint(self.tiny_model(), run, 1, 0.5, labels)
        payload = torch.load(path, weights_only=True)
        self.assertEqual(payload['run']['seed'], 42)
        self.assertEqual(payload['run']['config_file'], 'config.yaml')

    def test_all_real_experiment_paths_save_even_when_checkpoint_dir_is_none(self):
        for encoder, freeze in [('scratch', False), ('resnet18', True), ('resnet18', False)]:
            with self.subTest(encoder=encoder, freeze=freeze):
                run = self.run_config(encoder, freeze)
                model = (self.tiny_model() if encoder == 'scratch' else
                         build_classifier(encoder, num_classes=5, weights=None, freeze=freeze))
                with patch.object(experiment, 'build_model', return_value=model), \
                        patch.object(experiment, 'pick_device', return_value=torch.device('cpu')), \
                        patch.object(experiment, 'build_dataloaders', return_value=self.loaders), \
                        patch('src.training.trainer.train_one_epoch', return_value=(1., 0.2)), \
                        patch('src.training.trainer.evaluate', side_effect=[{'loss': 1., 'accuracy': 0.7},
                                                                           {'loss': 1., 'accuracy': 0.5}]):
                    result = experiment.train_run(run, self.data, checkpoint_dir=None)
                self.assertTrue(Path(result['checkpoint_path']).is_file())
                self.assertEqual(result['run'], run)
                self.assertEqual(result['best_epoch'], 1)
                table = comparison_table([result])
                self.assertEqual(table.loc[run['name'], 'best_val_accuracy'], 0.7)

    def test_transfer_resume_tuning_trials_and_retraining_track_checkpoints(self):
        model = self.tiny_model()
        config = {**self.run_config(), 'final_epochs': 2}
        with patch.object(experiment, 'build_model', return_value=model), \
                patch.object(experiment, 'pick_device', return_value=torch.device('cpu')), \
                patch.object(experiment, 'build_dataloaders', return_value=self.loaders), \
                patch('src.training.trainer.train_one_epoch', return_value=(1., 0.2)), \
                patch('src.training.trainer.evaluate', return_value={'loss': 1., 'accuracy': 0.6}):
            def train_fn(run, **kwargs):
                return experiment.train_run(run, self.data, **kwargs)

            run = self.run_config()
            result = run_and_save(run, train_fn, self.root / 'transfer_runs')
            skipped = run_and_save(run, train_fn, self.root / 'transfer_runs')
            self.assertEqual(result['checkpoint_path'], skipped['checkpoint_path'])
            objective = make_objective(config, train_fn)
            params = [{'lr': lr, 'weight_decay': 0., 'dropout': 0.3} for lr in [0.001, 0.002]]
            trials = run_search('grid', params, objective, self.root / 'trials.csv')
            self.assertNotEqual(trials.iloc[0]['checkpoint_path'], trials.iloc[1]['checkpoint_path'])
            resumed = run_search('grid', params, objective, self.root / 'trials.csv')
            self.assertEqual(list(resumed['checkpoint_path']), list(trials['checkpoint_path']))
            comparison = compare_searches(trials, 'val_accuracy', ['lr', 'weight_decay', 'dropout'])
            self.assertEqual(comparison.loc['grid', 'best_val_accuracy'], 0.6)
            with patch('src.tuning.best_params', return_value={'lr': 0.001, 'weight_decay': 0., 'dropout': 0.3}):
                retrain_best(config, train_fn, self.root)
            tuned = json.loads((self.root / 'transfer_runs/scratch_tuned.json').read_text())
            self.assertTrue(Path(tuned['checkpoint_path']).is_file())
            self.assertEqual(len(list(self.directory.glob('*.pth'))), 4)

    def test_cli_uses_root_models_for_limited_runs_and_rejects_external_locations(self):
        args = SimpleNamespace(dry_run=False, limit=2, table_dir=None, fig_dir=None, checkpoint_dir=None)
        original_cwd = Path.cwd()
        # Use a real cwd change, restored even on assertion failures.
        from contextlib import chdir
        notebooks = self.root / 'notebooks'
        notebooks.mkdir()
        with chdir(notebooks):
            tables, figures, directory = output_dirs(args)
        self.assertEqual(directory, self.directory)
        self.assertEqual(tables, Path('outputs/dry_run'))
        self.assertEqual(figures, Path('outputs/dry_run'))
        self.assertEqual(Path.cwd(), original_cwd)
        with self.assertRaisesRegex(ValueError, 'inside'):
            checkpoints.checkpoint_directory(self.root / 'outside')

    def test_invalid_metadata_and_zero_epochs_fail_before_training(self):
        model = self.tiny_model()
        optimizer = torch.optim.Adam(model.parameters())
        with patch('src.training.trainer.train_one_epoch') as train:
            with self.assertRaisesRegex(ValueError, 'epochs'):
                fit(model, self.loaders, nn.CrossEntropyLoss(), optimizer, 'cpu', 0)
            with self.assertRaisesRegex(ValueError, 'class_labels'):
                fit(model, self.loaders, nn.CrossEntropyLoss(), optimizer, 'cpu', 1, class_labels=['a'])
            train.assert_not_called()
        self.assertEqual(list(self.directory.glob('*.pth')), [])

    def test_synthetic_dry_run_does_not_save_random_weights(self):
        with patch.object(experiment, 'build_classifier', return_value=self.tiny_model()):
            result = experiment.synthetic_run(self.run_config(), checkpoint_dir=self.directory)
        self.assertNotIn('checkpoint_path', result)
        self.assertEqual(list(self.directory.glob('*.pth')), [])


if __name__ == '__main__':
    unittest.main()
