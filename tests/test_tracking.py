"""Read actual TensorBoard events to verify tracking without retraining bird models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
import torch
from torch import nn

from src import experiment
from src.models.factory import build_classifier
from src.training import checkpoints, tracking
from src.training.trainer import fit


class TrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for target in [patch.object(tracking, 'LOGS_DIR', self.root / 'logs'),
                       patch.object(checkpoints, 'MODELS_DIR', self.root / 'models')]:
            target.start()
            self.addCleanup(target.stop)
        self.model = build_classifier('scratch', num_classes=5, widths=(4, 8))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.run = {'name': 'trial', 'encoder': 'scratch', 'lr': 0.001, 'dropout': 0.3,
                    'freeze': False, 'image_size': 32}
        self.row = {'epoch': 1, 'train_loss': 1.2, 'val_loss': 1.4,
                    'train_accuracy': 0.6, 'val_accuracy': 0.4}

    def events(self, directory):
        return EventAccumulator(str(directory), size_guidance={'scalars': 0, 'tensors': 0}).Reload()

    def text(self, events, name):
        return events.Tensors(name + '/text_summary')[-1].tensor_proto.string_val[0].decode()

    def test_shared_fit_records_curves_hparams_and_best_checkpoint(self):
        with patch('src.training.trainer.train_one_epoch', side_effect=[(1.2, 0.6), (1.1, 0.7)]), \
                patch('src.training.trainer.evaluate', side_effect=[{'loss': 1.4, 'accuracy': 0.4, 'top5_accuracy': 1.},
                                                                   {'loss': 1.3, 'accuracy': 0.3, 'top5_accuracy': 1.}]):
            result = fit(self.model, {'train': [], 'validation': []}, nn.CrossEntropyLoss(),
                         self.optimizer, 'cpu', 2, run_config=self.run)
        events = self.events(result['tensorboard_dir'])
        self.assertEqual([point.step for point in events.Scalars('loss/train')], [1, 2])
        self.assertAlmostEqual(events.Scalars('accuracy/validation')[0].value, 0.4)
        self.assertEqual(len(events.Scalars('accuracy/validation_top5')), 2)
        self.assertAlmostEqual(events.Scalars('learning_rate/group_0')[0].value, 0.001)
        self.assertIn('"image_size": 32', self.text(events, 'run/config'))
        self.assertEqual(self.text(events, 'run/checkpoint'), result['checkpoint_path'])
        self.assertTrue(Path(result['checkpoint_path']).is_file())
        self.assertEqual(self.text(events, 'run/status'), 'completed')
        record = json.loads((Path(result['tensorboard_dir']) / 'run.json').read_text())
        self.assertEqual(record['best_epoch'], 1)
        self.assertEqual(record['completed_epochs'], 2)
        self.assertEqual(record['checkpoint_path'], result['checkpoint_path'])
        hparams = [HParamsPluginData.FromString(content)
                   for content in events.PluginTagToContent('hparams').values()]
        sessions = [value.session_start_info for value in hparams if value.HasField('session_start_info')]
        self.assertEqual(len(sessions), 1)
        self.assertAlmostEqual(sessions[0].hparams['lr'].number_value, 0.001)
        self.assertEqual(sessions[0].hparams['encoder_kwargs.widths'].string_value, '[4, 8]')
        self.assertAlmostEqual(events.Scalars('hparams/best_val_accuracy')[0].value, 0.4)

    def test_interrupted_and_failed_runs_keep_completed_curves(self):
        for exception, status in [(KeyboardInterrupt(), 'interrupted'), (RuntimeError('training failed'), 'failed')]:
            with self.subTest(status=status):
                with patch('src.training.trainer.train_one_epoch', side_effect=[(1.2, 0.6), exception]), \
                        patch('src.training.trainer.evaluate', return_value={'loss': 1.4, 'accuracy': 0.4}):
                    with self.assertRaises(type(exception)):
                        fit(self.model, {'train': [], 'validation': []}, nn.CrossEntropyLoss(),
                            self.optimizer, 'cpu', 2, run_config=self.run)
        records = [json.loads(path.read_text()) for path in (self.root / 'logs').glob('*/run.json')]
        self.assertEqual({record['status'] for record in records}, {'interrupted', 'failed'})
        for directory in (self.root / 'logs').iterdir():
            events = self.events(directory)
            self.assertEqual(len(events.Scalars('loss/train')), 1)
            self.assertIn(self.text(events, 'run/status'), {'interrupted', 'failed'})
            self.assertNotIn('run/checkpoint/text_summary', events.Tags()['tensors'])
        self.assertEqual(list((self.root / 'models').glob('*.pth')), [])

    def test_same_settings_produce_separate_runs_and_test_metrics_append(self):
        directories = []
        for _ in range(2):
            with tracking.TensorBoardRun(self.run) as run:
                run.log_epoch(self.row)
                run.complete(1, 0.4, 'models/reference.pth')
            directories.append(run.log_dir)
        self.assertNotEqual(*directories)
        tracking.log_test_metrics(directories[0], {'test_accuracy': 0.75, 'test_f1_macro': 0.7}, 1)
        self.assertAlmostEqual(self.events(directories[0]).Scalars('test/accuracy')[0].value, 0.75)
        self.assertNotIn('test/accuracy', self.events(directories[1]).Tags()['scalars'])

    def test_import_preserves_source_and_is_idempotent_without_saving_weights(self):
        source = self.root / 'source'
        source.mkdir()
        path = source / 'trial.json'
        saved = {'run': self.run, 'history': [self.row], 'best_epoch': 1, 'best_val_accuracy': 0.4,
                 'checkpoint_path': '/teammate/models/actual.pth', 'minutes': 2.,
                 'test_accuracy': 0.7, 'device': 'cpu'}
        path.write_text(json.dumps(saved))
        original = path.read_bytes()
        directories = tracking.import_saved_runs([source])
        events = self.events(directories[0])
        self.assertAlmostEqual(events.Scalars('loss/train')[0].value, 1.2)
        self.assertAlmostEqual(events.Scalars('test/accuracy')[0].value, 0.7)
        self.assertEqual(self.text(events, 'run/checkpoint'), saved['checkpoint_path'])
        self.assertIn('historical:', self.text(events, 'run/source'))
        event_files = list(directories[0].glob('events.*'))
        self.assertEqual(tracking.import_saved_runs([source]), directories)
        self.assertEqual(list(directories[0].glob('events.*')), event_files)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list((self.root / 'models').glob('*.pth')), [])

    def test_import_skips_synthetic_and_rejects_missing_histories(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'synthetic.json').write_text(json.dumps({'device': 'synthetic'}))
        self.assertEqual(tracking.import_saved_runs([source]), [])
        (source / 'invalid.json').write_text(json.dumps({'run': self.run}))
        with self.assertRaisesRegex(ValueError, 'history'):
            tracking.import_saved_runs([source])
        with self.assertRaisesRegex(ValueError, 'No saved'):
            tracking.import_saved_runs([self.root / 'missing'])

    def test_real_runner_appends_held_out_metrics_to_its_training_run(self):
        run = {**self.run, 'seed': 42, 'optimizer': 'adam', 'weight_decay': 0., 'epochs': 1,
               'batch_size': 2, 'num_workers': 0}
        from datasets import ClassLabel, Dataset, DatasetDict, Features
        split = Dataset.from_dict({'label': [0, 1, 2, 3, 4]},
                                  features=Features({'label': ClassLabel(names=['a', 'b', 'c', 'd', 'e'])}))
        data = DatasetDict({name: split for name in ['train', 'validation', 'test']})
        test_result = {'labels': np.array([0, 1, 2, 3, 4]), 'predictions': np.array([0, 1, 2, 3, 0]),
                       'loss': 1., 'accuracy': 0.8, 'top5_accuracy': 1.}
        with patch.object(experiment, 'build_model', return_value=self.model), \
                patch.object(experiment, 'pick_device', return_value=torch.device('cpu')), \
                patch.object(experiment, 'build_dataloaders', return_value={'train': [], 'validation': [], 'test': []}), \
                patch('src.training.trainer.train_one_epoch', return_value=(1.2, 0.6)), \
                patch('src.training.trainer.evaluate', return_value={'loss': 1.4, 'accuracy': 0.4}), \
                patch('src.evaluation.classification.evaluate', return_value=test_result):
            result = experiment.train_run(run, data, test=True)
        self.assertAlmostEqual(self.events(result['tensorboard_dir']).Scalars('test/accuracy')[0].value, 0.8)
        self.assertIn('test/f1_macro', self.events(result['tensorboard_dir']).Tags()['scalars'])


if __name__ == '__main__':
    unittest.main()
