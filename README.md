## Grad-CAM (assignment point 7)
After preparing `data/processed` and obtaining trained scratch and transfer checkpoints:

```bash
python -m src.cli gradcam --checkpoints models/YOUR_SCRATCH_CHECKPOINT.pth models/YOUR_TRANSFER_CHECKPOINT.pth
```

Run from the project root with the virtual environment active. Checkpoints must use the
project's `run` + `state_dict` format. This command does not train models or download weights.
Use `--max-images 100` for a shorter scan, `--examples 3` for the number per category,
and `--split validation` during development. CPU is the default; `--device cuda` or
`--device mps` can be selected if available.

Figures and `report.json` are saved under `outputs/grad_cam/<checkpoint-name>/`.
Use distinct checkpoint filenames for distinct models. Correct and wrong examples show
the evaluation crop alongside its predicted-class heatmap. Missing categories are reported.
The report records only selected examples and the number scanned, not test accuracy.
See `notebooks/grad_cam.ipynb` for a walkthrough and an analysis template.
Write conclusions from real trained-model figures; random-model checks are only software tests.

Verify Grad-CAM with `python -m unittest discover -s tests -v`.

The CLI also creates `outputs/grad_cam/comparison.html`, a portable gallery of both models.
For checkpoint troubleshooting, calculation details, and interpretation limits, see
[the Grad-CAM guide](docs/grad_cam.md).

## Automatic model checkpoints

Every completed training run now saves its best validation-accuracy weights automatically
in the project-root `models/` folder. This includes scratch and transfer runs, grid/random
search trials, best-setting retraining, the scratch notebook, and direct calls to shared `fit()`.
Synthetic `--dry-run` experiments do not save untrained weights.

Files use `<encoder>_<experiment>_<UTC timestamp>_<unique ID>_best.pth` names.
The full path is printed and included in experiment JSON and tuning CSV results as
`checkpoint_path`. No existing checkpoints are moved, deleted, or overwritten.
Weights remain excluded from Git; `models/.gitkeep` preserves the folder in the repository.

Load a saved model for evaluation or inference without downloading pretrained weights:

```python
from src.training.checkpoints import load_checkpoint

model, run = load_checkpoint("models/YOUR_CHECKPOINT.pth", device="cpu")
```

Replace placeholder filenames with the paths printed after training. The loader also
accepts previously saved `run`/`state_dict` `.pt` files. See the
[checkpoint guide](docs/checkpoints.md) for the format, failure behavior, and direct `fit()` usage.

## TensorBoard experiment tracking (section 5.7)

Install the updated dependencies in your virtual environment:

```bash
python -m pip install -r requirements.txt
```

All real runs using shared `fit()` now log automatically to unique folders in
`outputs/tensorboard/`. Scratch, frozen/fine-tuned transfer models, tuning trials,
best-setting retraining, and the scratch notebook share this integration.

Open the dashboard from a second terminal at the project root:

```bash
tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006
```

Visit <http://localhost:6006>. Compare loss/accuracy curves in **Scalars**, configurations
and best validation accuracy in **HParams**, and checkpoint paths/status in **Text**.
Weights continue to be saved in `models/`; logs contain their references.

Import existing training histories without retraining:

```bash
python -m src.cli tensorboard-import --runs-dir outputs/tables/transfer_runs outputs/my_training/tables/transfer_runs
```

Imported runs are labeled historical. Repeated imports skip identical results; synthetic
results are excluded. Old results that lack checkpoint references are marked as such.
See [the TensorBoard guide](docs/tensorboard.md) for tags, run comparison, and report evidence.
