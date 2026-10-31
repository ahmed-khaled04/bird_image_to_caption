# Grad-CAM workflow

## Files and responsibilities

- `src/evaluation/grad_cam.py`: gradient-based heatmap calculation, independent of plotting and disk I/O.
- `src/grad_cam.py`: checkpoint restoration, example selection, aligned overlays, JSON reports, HTML comparison.
- `src/cli.py`: the `gradcam` command.
- `notebooks/grad_cam.ipynb`: interactive walkthrough and analysis prompts.
- `tests/test_grad_cam.py`: mathematical reference, frozen-encoder coverage, state preservation, checkpoint and report checks.

## Run with your team's trained models

From the project root, activate `.venv`, then:

```bash
python -m src.cli gradcam --checkpoints outputs/checkpoints/scratch.pt outputs/checkpoints/resnet18_frozen.pt
open outputs/grad_cam/comparison.html
```

Replace the transfer checkpoint with your team's chosen model. Supported encoders are
scratch, ResNet18, ResNet50, and EfficientNet-B0. The checkpoint configuration determines
the architecture and image size. Weight loading is strict; no pretrained download is needed.

The CLI creates per-model PNG images and `report.json`, plus a portable HTML comparison
with embedded images. Rerunning replaces selected filenames and report/gallery files;
older PNGs may remain, so use the current JSON/gallery to identify the current selection.
Use a separate `--out-dir` for each experiment to retain previous results.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `data/processed` not found | Ask for the team's cleaned splits or run `python -m src.cli clean`. |
| Checkpoint not found | Obtain the real trained `.pt` files; dataset cache files are not checkpoints. |
| `run` or `state_dict` missing | Use checkpoints saved by this project's `train_run`; other notebook formats need conversion. |
| No correct or no wrong examples | Remove `--max-images` to scan more images; record missing categories honestly. |
| Blank heatmap | Positive weighted activation can be zero for the chosen class; do not manufacture a hotspot. |
| CUDA/MPS unavailable | Use the default `--device cpu`. |
| Slow scan | Try `--max-images 100 --examples 1` for a preview, then run a full scan. |
| Notebook import fails | Select the project environment and open the notebook from the project or notebooks folder. |

## How the explanation is calculated

For the selected class logit, compute gradients with respect to the encoder's final spatial
feature maps. Average gradients over height and width to get one weight per channel.
Sum the weighted maps, apply ReLU, resize to the model input, and normalize by the maximum.
A zero map remains zero. Predicted logits, rather than probabilities, are differentiated.

The input temporarily requires gradients so frozen encoder weights still yield differentiable
features. Model weights and parameter gradient buffers are not changed. Evaluation mode
prevents dropout randomness and BatchNorm updates, and original module modes are restored.
Hooks are removed even when a call fails. Use ordinary execution or `torch.no_grad()` around
the caller, not `torch.inference_mode()`.

## Assignment evidence

Include correct and wrong examples from both trained models with true/predicted labels.
Discuss bird parts versus background and compare observed localization. Explain that the
heatmaps describe the predicted class, even when it is wrong. Each map uses its own colour
scale; intensity cannot establish which model is more confident or more accurate.

The two models may select different images because their predictions differ. These are
illustrative examples, not a controlled same-image comparison or an accuracy benchmark.
Use figure references and actual observations in your written analysis. Do not infer
trained-model behavior from randomly initialized test models.
