# TensorBoard experiment tracking: assignment section 5.7

The project uses PyTorch's `torch.utils.tensorboard.SummaryWriter` to write real TensorBoard
event files. Configuration, curves, metrics, run status, and checkpoint references are
recorded automatically by shared `fit()`. Training hyperparameters, architectures, data
transforms, evaluation metrics, and model-selection rules are unchanged.

## Start the dashboard

From the project root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006
```

Open <http://localhost:6006>. Leave the server running while training in another terminal.
Stop the server with Ctrl+C. If port 6006 is occupied, use `--port 6007` and open that port.
In a notebook with the project kernel, `%load_ext tensorboard` followed by
`%tensorboard --logdir ../outputs/tensorboard` starts an embedded dashboard when the kernel
working directory is `notebooks/`; adjust the path if it starts at the project root.

## What every new training run logs

| Requirement | TensorBoard location |
| --- | --- |
| Full configuration | Text: `run/config` |
| Class labels in index order | Text: `run/class_labels` |
| Searchable/comparable hyperparameters | HParams: architecture/encoder, dropout, freeze, optimizer, learning rate, batch size, seed, image size, epochs and other saved settings |
| Training/validation loss | Scalars: `loss/train`, `loss/validation` |
| Training/validation accuracy | Scalars: `accuracy/train`, `accuracy/validation` |
| Existing validation top-5 accuracy | Scalars: `accuracy/validation_top5`, when available |
| Learning rates | Scalars: `learning_rate/group_0`, etc. |
| Timing | Scalars: `time/epoch_seconds`, `time/training_seconds` |
| Best validation result | HParams: `hparams/best_val_accuracy`, `hparams/best_epoch`; Scalars: `summary/*` |
| Checkpoint reference | Text: `run/checkpoint` |
| Run origin and status | Text: `run/source`, `run/status`; `run/error` when interrupted or failed |
| Test metrics, if explicitly evaluated by `train_run(test=True)` | Scalars: `test/accuracy`, `test/top5_accuracy`, `test/precision_macro`, `test/recall_macro`, `test/f1_macro` |

HParams is populated on successful training completion. Interrupted runs retain their
configuration in Text and their completed-epoch curves, with an explicit status. Partial
epochs are not reported as complete. Every epoch is flushed so curves appear during training.
The existing `evaluate` reporting command still writes its separate CSV/plot reports;
new TensorBoard test scalars are written by the shared experiment runner's `test=True` path.

Each training invocation gets a unique timestamp/UUID folder even when its experiment name
matches a previous invocation. The exact log directory is printed and returned as
`tensorboard_dir`; transfer JSON results and tuning CSV rows record it alongside `checkpoint_path`.
The event folder also contains `run.json` with the source configuration, status, completed
epoch count, best result, and checkpoint reference. Default logs are ignored by Git.

Weights remain in `models/`, with the existing unique `.pth` saving and Grad-CAM compatibility.
TensorBoard references the checkpoint instead of copying or serializing the entire model.
Scratch, frozen transfer, fine-tuned transfer, grid/random search, final retraining, and
notebook training all go through the same logger. Synthetic `--dry-run` scores are excluded.

## Import runs trained before TensorBoard was added

```bash
python -m src.cli tensorboard-import --runs-dir outputs/tables/transfer_runs outputs/my_training/tables/transfer_runs
```

Supply directories containing the project's saved training-result JSON files with `run`,
`history`, `best_epoch`, and `best_val_accuracy`. Importing reads these files and writes
TensorBoard logs under `outputs/tensorboard/imported/`; it never retrains or rewrites results
or checkpoints. Runs are marked `historical:<source JSON path>` in Text. Identical JSON content
is imported once, including when the same results were copied between directories.
Changing source results creates a separate import. Synthetic results marked `device=synthetic`
are skipped. An incomplete import requires a different log root for retrying.

Only recorded data is imported: epoch loss/accuracy, best result, configuration, optional
timing/test metrics, and checkpoint references when known. Missing historical checkpoint
references are explicitly marked as unrecorded. Per-epoch learning rates/top-5 values absent
from old histories are not reconstructed or invented. New runs record these values live.

## Compare and explain a run for the report

1. In HParams, select runs and compare encoder, freezing mode, optimizer settings, dropout,
   batch size, seed, and the best validation accuracy. Imported configurations show the
   settings available in their original JSON.
2. In Scalars/Time Series, overlay the selected runs' training and validation curves.
   Identify convergence speed and gaps between training and validation performance.
3. In Text, show the full configuration, source/status, and exact checkpoint reference.
   Clearly identify any historical result whose checkpoint is unavailable.
4. Include screenshots of HParams, the loss/accuracy curves, and a run's checkpoint reference.
   Base conclusions on the actual values; choose models with validation, not test results.

For example, compare a scratch model with a frozen ResNet18 and a fine-tuned ResNet18.
Explain their actual setting differences and observed results, rather than assuming that
transfer learning must win. A validation-selected checkpoint corresponds to its best epoch,
which may differ from the last point in the curves.

## Custom locations and shared utilities

Report-ready exports are also available without screenshots:

```bash
python -m src.cli tracking-export --runs-dir outputs/tables/transfer_runs outputs/my_training/tables/transfer_runs --out-dir outputs/tracking_evidence
```

Run the historical importer above first if the event files are absent. The exporter reads
actual event curves, Text, and HParams, checks configurations/epoch metrics against source
JSON, verifies available checkpoint references, and writes PNGs, CSV/JSON exports, a
checkpoint inventory with SHA-256, and a Markdown report. Original result JSON files are
not changed. Historical weights that are absent remain explicitly missing; exports do not
reconstruct them or silently attach another run's weights. See
`outputs/tracking_evidence/README.md` and `docs/reproducibility.md` for the current audit.

`transfer` and `tune` accept `--log-dir <root>`; weights still go to `models/`.
For direct/notebook use, `fit(..., tensorboard_dir=<root>)` sets the root. The historical
import command also supports `--log-dir`. Point TensorBoard at the same root when launching.
Existing completed-run resume behavior is unchanged: skipped runs produce no new training
events. Use the importer to view their previously recorded histories.

Implementation: `src/training/tracking.py`; integration: `src/training/trainer.py`,
`src/experiment.py`, `src/tuning.py`, and `src/cli.py`. Tests read actual event files,
including the HParams plugin data, with temporary fixtures and mocked epoch work:

```bash
python -m unittest discover -s tests -v
```

API reference: [PyTorch SummaryWriter documentation](https://docs.pytorch.org/docs/stable/tensorboard.html).
