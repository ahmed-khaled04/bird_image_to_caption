# Automatic model checkpoints

## Storage and naming

New checkpoints are always stored under `models/` at the project root, regardless of
the current working directory. The directory is created automatically and tracked with
an empty `.gitkeep`; the existing `*.pth` ignore rule keeps weights out of commits.

Each completed call to `src.training.trainer.fit()` writes one new file:

```text
models/<encoder>_<experiment>_<UTC timestamp>_<unique run ID>_best.pth
```

The timestamp includes microseconds, and a UUID distinguishes repeated experiment names
and simultaneous runs. Filename identifiers are sanitized. The checkpoint is the state
with the highest validation accuracy, using the existing strict improvement comparison;
ties keep the earlier epoch. The last epoch is not separately saved.

## Covered workflows

- CLI `transfer`, including scratch, frozen feature extractors, and fine-tuning.
- Every real grid-search and random-search trial from `tune`.
- `tune --retrain-best`.
- `notebooks/scratch_cnn.ipynb`, now using shared training/evaluation functions.
- Future classifier experiments using shared `fit()`, with no manual saving code.

Experiment JSON and tuning CSV files include `checkpoint_path`. Transfer and search resume
behavior is unchanged: already-completed runs are skipped and do not create another checkpoint.
Historical results that lack checkpoints are not retroactively populated or retrained.
`--limit` runs also save weights; their configuration and unique filename distinguish them.
`--dry-run` only fabricates scores and therefore saves no weights.

`--checkpoint-dir` remains available for subdirectories inside `models/`, such as
`--checkpoint-dir models/experiment_group`. Locations outside root `models/` are rejected.
This changes new-save destinations, while old checkpoints can still be loaded from any path.

## Format and reconstruction

```python
checkpoint = {
    "state_dict": ...,           # CPU tensors from the best validation state
    "run": {...},                # plain values, including reconstruction/training configuration
    "epoch": 7,                  # example: the selected best epoch, starting at 1
    "validation_metric": 0.81,   # example: validation accuracy at that epoch
    "metric_name": "val_accuracy",
    "best_val_accuracy": 0.81,   # compatibility alias
    "class_labels": [...],       # in model-output-index order
    "run_id": "...",
    "created_at": "...",         # UTC ISO timestamp
}
```

`run` preserves the training configuration plus architecture, encoder, class count, dropout,
freezing mode, image size, and any encoder constructor arguments needed for reconstruction.
The factory attaches reconstruction metadata without changing the architecture or weights.
For direct `fit()` calls, optimizer defaults, epoch count, loader settings, and class labels
are collected when available. If class names are unavailable, labels default to numeric
strings. Supply `class_labels` explicitly for custom datasets. Standard evaluation crop
size is inferred from the validation transform; supply `image_size` in `run_config` when
using custom transforms (the fallback is 224).

The payload contains no serialized model objects or optimizer states. It is suitable for
inference/evaluation and reconstruction, rather than exact mid-training optimizer resumption.

```python
import torch
from src.training.checkpoints import load_checkpoint

checkpoint = torch.load("models/YOUR_CHECKPOINT.pth", weights_only=True, map_location="cpu")
model, run = load_checkpoint("models/YOUR_CHECKPOINT.pth", device="cpu")
# The loader builds the architecture through build_classifier(), uses weights=None for
# transfer encoders, strictly loads state_dict, moves to the device, and sets eval().
```

The existing `src.grad_cam.load_checkpoint` import delegates to this same loader. Older
`.pt` files containing `run` and `state_dict` still load without the new metadata fields.
Grad-CAM calculations and dataset preprocessing are unchanged.

## Shared training usage

```python
from src.training.trainer import fit

trained = fit(model, loaders, loss_fn, optimizer, device, epochs,
              run_config={"name": "my_experiment", "image_size": 224, "seed": 42},
              class_labels=species)
print(trained["checkpoint_path"])
model.load_state_dict(trained["best_state"])
```

Use a classifier from the existing model factory. Existing positional `fit()` calls remain
valid and receive automatic saving. The utility still returns the same history and best-state
fields, plus `checkpoint_path`; it does not alter the in-memory final-epoch model. The
experiment runner and notebook restore the best state as before.

## Failed and interrupted runs

Saving happens after a successful training loop, before optional test evaluation. An
interruption during training creates no new checkpoint and leaves previous runs untouched.
Serialization uses a temporary file in `models/`; only a completed, flushed file is published
under its unique `.pth` name. Atomic publication refuses to replace an existing filename,
including in a collision. Normal write failures remove the temporary file. A forced process
kill may leave a hidden `.checkpoint-*.tmp`, which is not a usable checkpoint.

Tests use temporary directories, tiny randomly initialized fixtures, and mocked epoch work.
No real bird-model retraining or pretrained downloads are needed:

```bash
python -m unittest discover -s tests -v
```
