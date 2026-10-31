# Milestone 1 reproducibility verification

Reviewed on 31 October 2026, on branch `feat/classification-reproducibility`.
The [README](../README.md) contains setup and exact training, evaluation, inference,
Grad-CAM, and tracking commands. `requirements-tested.txt` captures the tested package
versions on Python 3.12.15 / macOS Apple Silicon. Initial dataset downloading and the
full original training experiments were not repeated in this verification.

## Executed checks

| Check | Scope | Result / evidence |
|---|---|---|
| Automated regression suite | Actual tensor/model restoration, gradients, tracking event files, identity checks, inference probabilities, and missing historical weights | 30 tests passed with `python -m unittest discover -s tests -v` |
| Real training | `configs/smoke.yaml`; first eight images of each split; one scratch epoch; MPS | Training, validation, unique checkpoint save, JSON/CSV/plots, and TensorBoard logging completed; `outputs/reproducibility/smoke_training/` |
| Scratch validation | Full 898-image validation split; local 30-epoch checkpoint at best epoch 29 | Accuracy 31.5145%; `outputs/reproducibility/evaluation/tables/validation_comparison.csv` |
| EfficientNet validation | Full 898-image validation split; local fine-tuned checkpoint at best epoch 10 | Accuracy 80.6236%; same table |
| Single-image CLI | Included validation image 0; CPU; five predictions | Spotted Catbird ranked first at 54.6874%; `outputs/reproducibility/prediction.json` |
| Grad-CAM CLI | Both trained local checkpoints; validation; up to 100 scanned images; MPS | Three correct and three wrong figures per model; `outputs/reproducibility/gradcam/comparison.html` |
| Historical tracking export | All seven team records plus two local records | Nine configurations, histories, and best metrics match actual imported TensorBoard events; `outputs/tracking_evidence/` |
| Weight audit | Recorded references, settings, best epoch/score, strict architecture restoration, SHA-256 | Two local references verified; seven original team checkpoints unavailable |

The real training smoke run is a software/pipeline check, not a full-data result or a tuning
trial. Its weights are separately named `scratch_smoke` and excluded from the nine historical
performance records. They remain local and Git-ignored like all model tensors.

## Source identity and checkpoint resolution

- Evaluation uses the result JSON's `checkpoint_path`, including relocation by exact basename.
- Every saved run setting must match the checkpoint, while extra reconstruction metadata is allowed.
- New checkpoints' best epoch and validation metric must match the result record.
- Class count and recorded label order must match the evaluation dataset.
- Legacy `<run-name>.pt` / `.pth` lookup remains supported when no reference was recorded.
- Missing explicit references do not fall back to a different run-name file.
- Multiple legacy candidates are rejected; an explicit reference is required to disambiguate.
- An enriched `.pth` is restored through the shared loader with pretrained downloads disabled.

Metadata consistency protects against accidental mismatches. It is not proof that an
unreferenced file belongs to a historical run. The audit therefore does not attach the
local reruns to the original records just because their settings or best scores overlap.

## Original weights still required

The local search covered the project, Documents, Downloads, and Desktop, including common
checkpoint extensions. Git history contains no committed `.pt` or `.pth` weights. Connected
Drive discovery did not identify a project model location; that connector's search is scoped
to supported document types and cannot establish that no model binaries exist in Drive.

The following original records still need their original checkpoints:

1. `scratch`
2. `resnet18_frozen`
3. `resnet18_finetune`
4. `resnet50_frozen`
5. `resnet50_finetune`
6. `efficientnet_b0_frozen`
7. `efficientnet_b0_finetune`

These are the files under `outputs/tables/transfer_runs/`. The local records under
`outputs/my_training/tables/transfer_runs/` are separately verified. Their epoch histories
and training durations differ from the team records, including EfficientNet even though
its best validation accuracy matches.

Obtain the original model files and provenance from the training environment/owner, copy
them into `models/`, and add `checkpoint_path` to the corresponding original JSON. Keep the
original history; do not replace it with invented curves or a new run. Reimport TensorBoard
and rerun `tracking-export`. If originals are permanently lost, document that limitation and
record any replacement training as a new experiment with its own history, weights, and ID.

## Evidence boundaries

The original team test accuracy (77.08%) remains in `outputs/tables/evaluation/`; this work
repeated validation of the two local checkpoints and did not claim to reproduce that test
evaluation. Original reports are preserved, and verification results use a separate folder.

Tracking exports read actual event files from historical imports and verify them against
source histories. They include only recorded values. Missing original checkpoints remain
explicitly missing. Raw TensorBoard logs, datasets, and weights are Git-ignored; report
exports, hashes, source, configurations, tests, and verification figures are committed.
