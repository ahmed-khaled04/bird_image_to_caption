# Milestone 1: outline and completion checklist

Status reviewed: 31 October 2026. Based on the assignment PDF, current source code, saved outputs, and Git history in this checkout. Work stored elsewhere is not verified here.

Milestone 1 covers Task 1.1 (data understanding and EDA) and Task 1.2 (classification and encoder training). The assignment gives 1 November as its deadline and requires an LMS submission even though this milestone is not graded separately. Colorization and caption generation belong to Milestone 2.

## Suggested report outline

1. **Introduction and scope**
   - Explain the full three-task project and the role of the shared visual encoder.
   - State that this milestone develops and evaluates the bird classifier and encoder.
2. **Dataset and exploratory data analysis**
   - Describe CUB-200-2011, 200 species, 11,788 images, and ten raw captions per image.
   - Present class distribution, image size/mode/quality analysis, and duplicate findings.
   - Present caption lengths, vocabulary, duplicates, empty lines, and spelling problems.
   - Explain how these observations influenced preprocessing and modeling.
3. **Cleaning, preprocessing, and data splits**
   - Explain caption splitting, normalization, typo corrections, short-caption filtering, and deduplication.
   - Explain removing eight duplicate training images while retaining the original test set.
   - Report the current 5,088 / 898 / 5,794 train / validation / test split and stratification.
   - Explain RGB conversion, resizing/cropping, normalization, and leakage prevention.
4. **Augmentation strategy**
   - Show original and augmented examples.
   - Justify small rotations, resized crops, horizontal flips, and mild color jitter.
   - Explain why augmentation is applied only to training images.
5. **Model architectures**
   - Describe the scratch CNN: stem, residual stages, and feature maps.
   - Explain global average pooling, dropout, and the 200-class output layer.
   - Describe ResNet-18, ResNet-50, and EfficientNet-B0.
   - Explain frozen encoders versus fine-tuning and the reusable encoder interface.
6. **Training setup and experiment tracking**
   - Report loss, optimizer, learning rates, batch size, epochs, seed, hardware, and parameter counts.
   - Explain selecting the best epoch using validation accuracy.
   - Include training/validation curves and a run comparison table.
   - Explain the custom JSON/CSV experiment records and checkpoint references; include exports/screenshots as evidence.
7. **Hyperparameter tuning**
   - List learning-rate, weight-decay, and dropout search spaces.
   - Explain grid and random search and their equal trial budgets.
   - Report actual trial results, best validation scores, settings, and time.
   - Compare baseline and tuned results, including unsuccessful tuning if applicable.
8. **Evaluation and final classifier selection**
   - Define accuracy, top-five accuracy, macro precision, recall, F1, and confusion matrices.
   - Compare scratch, frozen, and fine-tuned setups using validation results.
   - Explain the EfficientNet-B0 selection and report its held-out test metrics.
   - Discuss species confusions and limitations; distinguish validation from test results.
9. **Grad-CAM explainability**
   - Briefly explain the calculation.
   - Include trained scratch and transfer examples with correct and incorrect predictions.
   - Interpret bird-part versus background focus using the actual figures.
   - State qualitative interpretation limits.
10. **External benchmarking**
    - Compare classification results with a published result or open-source model.
    - Check dataset, split, metric, pretraining, and other protocol differences.
    - Cite sources and explain limitations of the comparison. This is a project-wide requirement; doing the classification comparison now supports Task 1.
11. **Failures, limitations, and next steps**
    - Discuss weak scratch performance, model errors, and experiment limitations.
    - Explain which encoder will be considered for Milestone 2 and why.
12. **Team contributions and references**
    - Describe meaningful contributions, supported by commits.
    - Cite datasets, pretrained models, libraries, methods, and benchmarking sources.

## Completion checklist

**How to read this:** `[x]` means the specific implementation or output is evidenced; `[ ]` means unfinished or not verified here. Code completion and experiment completion are separate tasks. Names indicate evidenced contributors, not exclusive ownership. Pending tasks have no assigned owner unless the team has confirmed one.

**Contributor names:** Ahmed Khaled (`ahmed-khaled04`), Mohamed Dahy (`Mohamed-Dahy`), Saifeldin Tamer Mohamed Elsayes (`Saifeldinsais`), and Yousef Sameh (`Yousef Sameh` in local commits). The `YoussifSameh2004` merge account is not counted as authorship of all merged code. Attribution comes from author commits; Yousef's local reruns and Grad-CAM interpretation are also evidenced in the project chats and local artifacts. Uncommitted outputs do not establish a separate teammate's authorship.

### Task 1.1: data understanding and preparation

- [x] Load the CUB train/test datasets. **Worked on by: Mohamed Dahy.** Evidence: `7104f33`, `src/data/loader.py`.
- [x] Analyze image sizes, modes, integrity, and duplicates. **Worked on by: Mohamed Dahy.** Evidence: `9f2c02b`, `src/data/image_stats.py`, image figures/tables.
- [x] Analyze class distribution and imbalance. **Worked on by: Mohamed Dahy.** Evidence: `98b0a7d`, `src/data/class_stats.py`, `outputs/tables/class_counts.csv`.
- [x] Analyze caption lengths, vocabulary, formatting, duplicates, and typos. **Worked on by: Ahmed Khaled and Mohamed Dahy.** Ahmed's notebook/cleaning analysis: `88175c1`; Mohamed's CLI port and vocabulary coverage: `26c8408`.
- [x] Save EDA plots and tables. **Worked on by: Ahmed Khaled and Mohamed Dahy; local reruns by Yousef Sameh.** Evidence: `outputs/figures/`, `outputs/tables/`, `outputs/my_cleaning/`.
- [x] Implement caption splitting and cleaning. **Worked on by: Ahmed Khaled, with fixes by Mohamed Dahy.** Evidence: `142f9ce`, `bd0a271`, `68f30f6`, `113375f`.
- [x] Remove identified duplicate training images before splitting. **Worked on by: Mohamed Dahy.** Evidence: `113375f`, `src/cleaning.py`.
- [x] Save stratified train/validation splits and preserve the original test set. **Worked on by: Ahmed Khaled, with duplicate-handling fixes by Mohamed Dahy.** Evidence: `bd0a271`, `113375f`, `data/processed/`, `outputs/tables/split_class_counts.csv`.
- [x] Implement RGB conversion, normalization, and evaluation transforms. **Worked on by: Ahmed Khaled.** Evidence: `76527cb`, `src/data/transforms.py`.
- [x] Implement training augmentation and save examples. **Worked on by: Ahmed Khaled; reproducibility fixes by Mohamed Dahy.** Evidence: `aac7cc1`, `113375f`, `outputs/figures/augmentation_examples.png`.
- [ ] Write EDA findings and preprocessing/augmentation justifications in the report. **Owner: unassigned.**

### Task 1.2: classification and transfer learning

- [x] Implement a custom CNN encoder and 200-class classifier. **Worked on by: Ahmed Khaled.** Evidence: `4b59566`, `src/models/encoder.py`, `src/models/classifier.py`.
- [x] Implement the shared training loop and validation-based best-epoch selection. **Worked on by: Saifeldin Tamer; training integration by Mohamed Dahy.** Evidence: `02997df`, `5d6acb8`, `src/training/trainer.py`.
- [x] Save a 30-epoch scratch experiment record. **Worked on by: Saifeldin Tamer; additional local run by Yousef Sameh.** Evidence: `02997df`, `outputs/tables/transfer_runs/scratch.json`, `outputs/my_training/tables/transfer_runs/scratch.json`.
- [x] Implement pretrained ResNet-18, ResNet-50, and EfficientNet-B0 encoders. **Worked on by: Mohamed Dahy.** Evidence: `01a2544`, `src/models/pretrained.py`.
- [x] Implement frozen and fine-tuned training modes. **Worked on by: Mohamed Dahy.** Evidence: `01a2544`, `a41bd3f`, `src/transfer.py`.
- [x] Save results for all six pretrained setups. **Worked on by: Saifeldin Tamer.** Evidence: `02997df`, `f1e671b`, six transfer JSON records in `outputs/tables/transfer_runs/`.
- [x] Save scratch-versus-transfer comparison tables and learning curves. **Worked on by: Mohamed Dahy (comparison pipeline) and Saifeldin Tamer (training results); local two-model comparison by Yousef Sameh.** Evidence: `a41bd3f`, `02997df`, `outputs/tables/transfer_comparison.csv`, `outputs/my_training/`.
- [x] Identify fine-tuned EfficientNet-B0 as the best validation setup among these runs. **Worked on by: Saifeldin Tamer; reproduced locally by Yousef Sameh.** Evidence: saved validation comparisons; validation accuracy **80.62%**.
- [ ] Explain architectures, settings, comparisons, and final selection in the report. **Owner: unassigned.**

### Required hyperparameter tuning

- [x] Implement grid search and random search with learning-rate, weight-decay, and dropout spaces. **Worked on by: Mohamed Dahy.** Evidence: `08a6fb9`, `5d6acb8`, `src/training/search.py`, `configs/tuning.yaml`.
- [x] Implement equal trial budgets, CSV trial records, comparison plots, and best-setting retraining. **Worked on by: Mohamed Dahy.** Evidence: `08a6fb9`, `5d6acb8`, `src/tuning.py`.
- [ ] Complete real grid-search trials and retain results. **Owner: unassigned.**
- [ ] Complete real random-search trials and retain results. **Owner: unassigned.**
- [ ] Compare methods using actual scores, trial counts, and elapsed time. **Owner: unassigned.**
- [ ] Train/evaluate the chosen tuned setting and compare it with its untuned baseline. **Owner: unassigned.**
- [ ] Include search settings, results, and conclusions in the report. **Owner: unassigned.**

No real tuning result files were found in this checkout. Implemented search code and synthetic dry runs do not complete the required tuning experiments.

### Evaluation

- [x] Implement accuracy, top-five accuracy, precision, recall, F1, and confusion reporting. **Worked on by: Saifeldin Tamer.** Evidence: `02997df`, `f1e671b`, `src/evaluation/classification.py`.
- [x] Save per-class metrics and confusion outputs for the seven classifier setups. **Worked on by: Saifeldin Tamer.** Evidence: `f1e671b`, `outputs/tables/evaluation/`, `outputs/figures/evaluation/`.
- [x] Save final test results for fine-tuned EfficientNet-B0. **Worked on by: Saifeldin Tamer.** Evidence: `f1e671b`, `outputs/tables/evaluation/final_test_metrics.csv`.
- [ ] Explain each reported metric and interpret errors in the report. **Owner: unassigned.**
- [ ] Reconcile older notebook outputs with current experiment records; label different runs and splits clearly. **Owner: unassigned.**

Saved test scores: accuracy **77.08%**, top-five accuracy **95.17%**, macro F1 **76.78%**. These come from the saved team evaluation; the two local reruns are separate experiment records.

### Grad-CAM

- [x] Implement Grad-CAM for scratch and transfer encoders. **Worked on by: Yousef Sameh.** Evidence: `92d12dd`, `src/evaluation/grad_cam.py`.
- [x] Implement correct/incorrect example selection and report/gallery generation. **Worked on by: Yousef Sameh.** Evidence: `a12d999`, `6b9c88a`, `src/grad_cam.py`.
- [x] Add documentation, a notebook, and software tests. **Worked on by: Yousef Sameh.** Evidence: `1a09c21`, `8947b44`, `dce995e`.
- [x] Obtain actual trained scratch and transfer checkpoints. **Worked on by: Yousef Sameh (local training).** Evidence: two `.pth` files in `models/`, referenced by `outputs/my_training/tables/transfer_runs/`.
- [x] Generate heatmaps from both trained models. **Worked on by: Yousef Sameh.** Evidence: `outputs/my_gradcam/comparison.html` and two model-specific `report.json` files; each contains three correct and three wrong validation predictions, with 12 saved figures in total.
- [x] Review correct and incorrect predictions using the figures. **Worked on by: Yousef Sameh with assistant-supported review.** Evidence: interpretation in the project chat “Explain the team's project work” on 31 October; transfer activations overlap bird regions more consistently in the selected examples, while some scratch errors highlight background. This analysis still needs to be copied into the report.
- [ ] Include heatmaps and evidence-based conclusions in the report. **Owner: unassigned.**

### Tracking, checkpoints, CLI, and reproducibility

- [x] Use modular Python packages for data, models, training, evaluation, and plots. **Worked on by: Ahmed Khaled, Mohamed Dahy, Saifeldin Tamer, and Yousef Sameh.** Evidence: incremental author commits across `src/`.
- [x] Include requirements and experiment configuration files. **Worked on by: Ahmed Khaled and Mohamed Dahy, with later dependency updates by Yousef Sameh.** Evidence: `458915c`, transfer/tuning commits, tracking commits.
- [x] Record transfer-run hyperparameters, epoch metrics, hardware, and timing in JSON/CSV outputs. **Worked on by: Mohamed Dahy and Saifeldin Tamer.** Evidence: transfer runner/training commits and `outputs/tables/transfer_runs/`.
- [x] Implement automatic best-weight checkpoint saving and loading across training workflows. **Worked on by: Yousef Sameh.** Evidence: `dfe137d`, `6692da1`, `320cdfc`, `033fd0d`, `src/training/checkpoints.py`.
- [x] Retain local scratch and EfficientNet checkpoints with references in the corresponding local training records. **Worked on by: Yousef Sameh.** Evidence: `models/*.pth` and `outputs/my_training/tables/transfer_runs/*.json`. This does not establish checkpoint availability for all historical team runs.
- [x] Implement TensorBoard logging of hyperparameters, epoch metrics/curves, and checkpoint references for shared training. **Worked on by: Yousef Sameh.** Evidence: `4aad4b7`, `c063bdd`, `src/training/tracking.py`.
- [x] Implement historical-run import and comparison documentation; retain imported event files. **Worked on by: Yousef Sameh.** Evidence: `0a9760a`, `7f9a339`, `outputs/tensorboard/imported/`, `docs/tensorboard.md`. Historical imports cannot restore missing weights or references.
- [x] Expose EDA, cleaning, augmentation, transfer training, tuning, evaluation, Grad-CAM, and tracking import through the CLI. **Worked on by: Ahmed Khaled (cleaning/augmentation), Mohamed Dahy (EDA/transfer/tuning), Saifeldin Tamer (evaluation), and Yousef Sameh (Grad-CAM/tracking import and integration).** Evidence: `src/cli.py` and respective feature commits.
- [ ] Recover the seven original historical checkpoints and verify their provenance/references. **Recovery owner: unassigned; audit completed by Yousef Sameh.** `outputs/tracking_evidence/coverage.csv` verifies two local runs and identifies all seven missing originals. The local reruns have different histories and cannot replace the originals silently. See `docs/reproducibility.md`.
- [x] Update and verify evaluation with the current checkpoint naming scheme. **Worked on by: Yousef Sameh.** The evaluator resolves explicit/relocated `.pth` references and legacy `.pt` files, checks settings and best-epoch/score metadata, validates labels, and restores via the shared loader. Both local models passed full validation; evidence: `outputs/reproducibility/evaluation/`.
- [x] Verify/document classification training, evaluation, and Grad-CAM commands using actual checkpoint paths. **Worked on by: Yousef Sameh.** `README.md`, `docs/reproducibility.md`, and `requirements-tested.txt`; verified real eight-image training, full validation of both local checkpoints, and 12 regenerated Grad-CAM figures. Full original training was not repeated.
- [x] Add and verify a single-image classification inference CLI. **Worked on by: Yousef Sameh.** `python -m src.cli predict` uses checkpoint labels and deterministic evaluation transforms; verified ranked JSON predictions with a real trained EfficientNet checkpoint. Evidence: `src/predict.py`, `tests/test_inference.py`, `outputs/reproducibility/prediction.json`.
- [x] Complete the README with environment setup and exact training/inference instructions. **Worked on by: Yousef Sameh.** Includes data preparation, full and smoke training, evaluation, image prediction, actual weight paths, Grad-CAM, tracking/export commands, and explicit missing-weight limitations.
- [x] Export tracking evidence for the report and audit every saved run's coverage. **Worked on by: Yousef Sameh.** `python -m src.cli tracking-export` exports actual TensorBoard curves, configuration/HParams, source/weight hashes, and a coverage table. All nine saved histories/configurations/best metrics match events; two local checkpoints are verified and seven original references remain missing under the recovery item above. Evidence: `outputs/tracking_evidence/`.

### Benchmarking, report, and submission

- [ ] Add an external classification benchmark and explain split, metric, and pretraining differences. **Owner: unassigned.**
- [ ] Write the Milestone 1 report using the outline above and export it as PDF. **Owner: unassigned.**
- [ ] Include failures, limitations, team contributions, and references. **Owner: unassigned.**
- [ ] Check that figures, tables, run settings, and reported scores agree. **Owner: unassigned.**
- [ ] Prepare and upload the Milestone 1 work to the LMS by **1 November**. **Owner: unassigned.**

The assignment requires Milestone 1 submission through the LMS. Its `TeamXX.zip`, model Drive-link, and end-to-end video list appears under final deliverables; check any separate Milestone 1 LMS packaging instructions.

## Team contribution summary

| Contributor | Evidenced work |
|---|---|
| Ahmed Khaled | Project setup; caption EDA/cleaning/splitting; augmentation; evaluation transforms; scratch CNN/classifier and experiment notebook. |
| Mohamed Dahy | Dataset loader; image/class EDA; caption-analysis CLI; duplicate/cleaning fixes; pretrained encoders; transfer runner/comparison; grid/random tuning implementation and integration. |
| Saifeldin Tamer Mohamed Elsayes | Shared training loop; completed team classification experiments; validation comparisons; multi-metric/per-class evaluation; final test results. |
| Yousef Sameh | Grad-CAM implementation, gallery, tests, documentation and local results; automatic checkpoints; local scratch/EfficientNet reruns; TensorBoard tracking, historical import, tests and guide. |

This table records evidenced contributions. It is not a grading allocation, and local or remote work absent from the inspected evidence may need to be added by the team.

## Suggested remaining-work order

1. Assign owners and run both real tuning searches; compare scores, trial counts, and time, then evaluate the best tuned setting.
2. Resolve evaluation/checkpoint compatibility and missing historical checkpoint references.
3. Use the verified classification inference and setup/run instructions; obtain the missing original weights for historical-run reproduction.
4. Use the exported tracking evidence, add the classification benchmark, and assemble the report, including the existing Grad-CAM analysis.
5. Check consistency and submit through the LMS.

## Outside Milestone 1

Image colorization and caption generation belong to Milestone 2. Caption cleaning and grayscale inspection are preparation, not completion of those models.

## Evidence locations

- Assignment: `../bOa6rh-Deep Learning - Final Project.pdf` (sections 5-7).
- Dataset/split evidence: `outputs/tables/class_counts.csv`, `outputs/tables/split_class_counts.csv`, `data/processed/`.
- Seven-run comparison and records: `outputs/tables/transfer_comparison.csv`, `outputs/tables/transfer_runs/`.
- Local scratch/transfer runs: `outputs/my_training/tables/transfer_runs/` and `models/`.
- Evaluation: `outputs/tables/evaluation/` and `outputs/figures/evaluation/`.
- Tuning implementation/settings: `src/tuning.py`, `src/training/search.py`, `configs/tuning.yaml`.
- Actual Grad-CAM results: `outputs/my_gradcam/comparison.html`, model-specific `report.json` files and figures.
- Grad-CAM implementation/guide: `src/grad_cam.py`, `src/evaluation/grad_cam.py`, `docs/grad_cam.md`.
- Checkpoints: `src/training/checkpoints.py`, `docs/checkpoints.md`.
- Tracking: `src/training/tracking.py`, `outputs/tensorboard/imported/`, `docs/tensorboard.md`.
- CLI/evaluation integration: `src/cli.py`, `src/evaluation/report.py`.
- Contribution evidence: Git author history, referenced commit hashes, and project chats for local runs/figure review.

Initial status review checked file contents, saved outputs, and commit history. The reproducibility follow-up ran 30 passing tests, real limited-data training, full validation for both available local checkpoints, image prediction, Grad-CAM generation, and tracking export. Seven original historical checkpoints are still unavailable; see `docs/reproducibility.md` for verification scope and evidence.
