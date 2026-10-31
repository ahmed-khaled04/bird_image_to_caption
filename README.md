

## Grad-CAM (assignment point 7)
After preparing `data/processed` and obtaining trained scratch and transfer checkpoints:

```bash
python -m src.cli gradcam --checkpoints outputs/checkpoints/scratch.pt outputs/checkpoints/resnet18_frozen.pt
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
