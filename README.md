# From Pixels to Language: bird classification and shared encoder

Milestone 1 uses CUB-200-2011 to study the data, train a scratch CNN, compare pretrained
encoders, evaluate classification, and explain predictions with Grad-CAM. The final code
runs through `python -m src.cli`; notebooks contain experiments. Colorization and caption
models are Milestone 2 work.

## Setup

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.cli --help
```

`requirements-tested.txt` records the exact packages in the verified Python 3.12.15 macOS
Apple Silicon environment. On a matching platform, use
`python -m pip install -r requirements-tested.txt` instead of the general requirements.
Other platforms should use the general requirements and their appropriate PyTorch build.
Training automatically chooses CUDA, then Apple MPS, then CPU. Prediction defaults to CPU;
evaluation defaults to automatic selection. Use `--device cpu` for portable evaluation.

## Prepare data and inspect it

The loader downloads `Multimodal-Fatima/CUB_train` and `Multimodal-Fatima/CUB_test` into
`data/hf_cache/`. The first download requires internet access and disk space for the raw
cache and processed copies. Existing cached data can be reused.

```bash
python -m src.cli eda --part summary
python -m src.cli eda --part images
python -m src.cli eda --part classes
python -m src.cli eda --part captions
python -m src.cli clean --out-dir data/processed --val-fraction 0.15 --seed 42
python -m src.cli augment --seed 0
```

Run image EDA before cleaning so `outputs/tables/duplicate_pairs.csv` identifies training
images to drop. Cleaning splits/normalizes captions, removes identified duplicate training
images, stratifies the train/validation split, and preserves the original test set.
The reviewed data has 5,088 training, 898 validation, and 5,794 test images. Training uses
RGB conversion, augmentation, and ImageNet normalization. Evaluation uses deterministic
RGB conversion, resize, center crop, and the same normalization.

Datasets are Git-ignored. Rebuild `data/processed/` with `clean` on another machine.

## Train classifiers

Train all seven settings from `configs/transfer.yaml` into a fresh output directory:

```bash
python -m src.cli transfer --config configs/transfer.yaml --data-dir data/processed --num-workers 0 --table-dir outputs/new_training/tables --fig-dir outputs/new_training/figures
```

To train only the scratch baseline and fine-tuned EfficientNet:

```bash
python -m src.cli transfer --config configs/transfer.yaml --runs scratch efficientnet_b0_finetune --data-dir data/processed --num-workers 0 --table-dir outputs/new_training/tables --fig-dir outputs/new_training/figures
```

Completed records in the chosen output directory are skipped on subsequent invocations.
Use a fresh output directory for a genuinely new experiment. The training loop keeps the
best validation epoch, saves unique weights under `models/`, and records `checkpoint_path`
and `tensorboard_dir` in the result JSON. `num-workers 0` is suitable for local execution;
GPU notebook environments can use the worker count in the configuration.

A short **real training smoke check**, using eight images per split and one epoch:

```bash
python -m src.cli transfer --config configs/smoke.yaml --limit 8 --num-workers 0 --table-dir outputs/reproducibility/smoke_training/tables --fig-dir outputs/reproducibility/smoke_training/figures --log-dir outputs/tensorboard/smoke
```

This check validates the pipeline and saves real weights; its scores are not full-dataset
results. Synthetic `--dry-run` checks do not train or save weights. Neither kind of check
counts as the required hyperparameter tuning experiments.

To run real grid/random tuning and then retrain its best setting:

```bash
python -m src.cli tune --config configs/tuning.yaml --data-dir data/processed --num-workers 0
python -m src.cli tune --config configs/tuning.yaml --data-dir data/processed --num-workers 0 --retrain-best
```

These commands are provided for the remaining tuning work; real tuning was not run as
part of the reproducibility work. Search settings use validation, never test results.

## Recover trained weights

Model tensors are excluded from Git. Obtain the original `.pth` files separately and place
them under `models/`. This checkout has these two verified **local** runs:

```bash
SCRATCH_CHECKPOINT="models/scratch_scratch_20261031T154728232806Z_855324b0da32471da0b535be9d043dba_best.pth"
TRANSFER_CHECKPOINT="models/efficientnet_b0_efficientnet_b0_finetune_20261031T162003884412Z_9336253ff19d4132b4d5f6c5fe3343b4_best.pth"
```

Keep the names when copying them to another machine. The evaluator can resolve a recorded
absolute reference by its exact basename under `models/` or `--checkpoint-dir`. Newly
trained files have different timestamp/UUID names; use their result JSON references.

**The seven original team records in `outputs/tables/transfer_runs/` have no recovered
original checkpoints in this checkout.** Their settings and curves remain valid saved
records, but they cannot all be reevaluated without those weights. The two local runs
have different histories; do not attach their weights to the older histories.
See [the coverage audit](outputs/tracking_evidence/README.md) for every missing record,
verified checkpoint hashes, and recovery steps. No model download/share link is currently
recorded. See [checkpoint details](docs/checkpoints.md) for the serialization format.

## Evaluate saved runs

Reevaluate both available local runs on the full validation split:

```bash
python -m src.cli evaluate --data-dir data/processed --runs-dir outputs/my_training/tables/transfer_runs --runs scratch efficientnet_b0_finetune --num-workers 0 --device auto --table-dir outputs/reproducibility/evaluation/tables --fig-dir outputs/reproducibility/evaluation/figures
```

For freshly trained runs, replace `--runs-dir` with `outputs/new_training/tables/transfer_runs`.
The evaluator reads each record's `checkpoint_path`, checks saved settings and any available
best-epoch/score metadata, validates class-label order, and restores the architecture with
pretrained downloads disabled. Legacy `<run-name>.pt` files are supported in `models/`,
`outputs/checkpoints/`, or `--checkpoint-dir`; ambiguous or mismatched files fail explicitly.

After selecting a model using validation, optionally add
`--test-run efficientnet_b0_finetune` to evaluate that selected model on the held-out test
split. Do not select architectures or tune using test scores. Reports include per-class
metrics, confusion matrices/pairs, comparison figures, and `provenance.json`. The verification
here repeated full validation; it did not repeat the previously saved team test evaluation.

## Predict one image

No dataset download is needed. After assigning `TRANSFER_CHECKPOINT` above:

```bash
python -m src.cli predict --checkpoint "$TRANSFER_CHECKPOINT" --image outputs/reproducibility/validation_00000.png --top-k 5 --device cpu --output outputs/reproducibility/prediction.json
```

Replace `--image` with any local photograph. The included image is validation example 0,
a Spotted Catbird. JSON output contains ranked class IDs, species, and softmax probabilities.
These probabilities are model scores, not calibrated guarantees. Class names and crop size
come from the checkpoint. Legacy weights without labels require `--labels path/to/labels.json`,
a JSON list of class names in training label-index order. Incorrect label counts/order are
rejected when checkpoint metadata is available.

## Explain predictions with Grad-CAM

After assigning both checkpoint variables above:

```bash
python -m src.cli gradcam --checkpoints "$SCRATCH_CHECKPOINT" "$TRANSFER_CHECKPOINT" --data-dir data/processed --split validation --examples 3 --max-images 100 --device cpu --out-dir outputs/reproducibility/gradcam
```

This generates three correct and three incorrect examples per model when the scanned
prefix contains them, plus a portable `comparison.html`. The verified gallery is at
[outputs/reproducibility/gradcam/comparison.html](outputs/reproducibility/gradcam/comparison.html).
These selected examples do not measure overall accuracy. Colors are normalized independently
per image and maps explain the predicted class. Use `--device mps` or `cuda` when available.
See [Grad-CAM details](docs/grad_cam.md).

## Track and export experiments

New real training runs automatically log configurations, loss/accuracy curves, best validation
metrics, and checkpoint references. From a separate terminal:

```bash
tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006
```

Open <http://localhost:6006> and inspect Scalars, HParams, and Text. Import saved historical
histories and export report evidence without retraining:

```bash
python -m src.cli tensorboard-import --runs-dir outputs/tables/transfer_runs outputs/my_training/tables/transfer_runs
python -m src.cli tracking-export --runs-dir outputs/tables/transfer_runs outputs/my_training/tables/transfer_runs --out-dir outputs/tracking_evidence
```

The exports include accuracy/loss PNGs read from actual events, run settings, raw event
exports, a run-by-run coverage CSV, and verified checkpoint metadata/hashes. All nine saved
histories have matching configuration, curves, and best metrics; two have verified local
weights and seven lack their original weights. Imported historical logs cannot recover
unrecorded information. See [tracking instructions](docs/tensorboard.md) and
[report-ready exports](outputs/tracking_evidence/README.md).

## Verification and project status

```bash
python -m unittest discover -s tests -v
```

The latest verification passed **30 tests**, a real limited-data training run, full validation
for both available models, single-image prediction, trained-model Grad-CAM, and the tracking
coverage export. See [reproducibility results](docs/reproducibility.md) for exact scope and
[the milestone checklist](MILESTONE_1_CHECKLIST.md) for contributions and unfinished work.

The tested environment and seeds make the procedure repeatable; exact training scores can
vary with hardware, library versions, augmentation randomness, and numerical kernels.
