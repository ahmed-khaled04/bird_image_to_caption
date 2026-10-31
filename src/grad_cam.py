"""Checkpoint-based Grad-CAM reports, without training or downloading model weights."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk

from src.data.loader import class_names
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.evaluation.grad_cam import GradCAM
from src.models.factory import build_classifier


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    run = checkpoint["run"]
    kwargs = {} if run["encoder"] == "scratch" else {"weights": None, "freeze": run["freeze"]}
    model = build_classifier(run["encoder"], num_classes=checkpoint["state_dict"]["fc.weight"].shape[0],
                             dropout=run["dropout"], **kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), run


def generate_report(checkpoint, split, out_dir, examples=3, max_images=None, device="cpu", split_name="test"):
    """Select first correct/wrong examples in dataset order; explain predicted logits.

    Counts describe only the scanned prefix, not full test accuracy. Missing categories
    are reported explicitly rather than substituting examples with invented outcomes.
    """
    if examples < 1 or (max_images is not None and max_images < 1):
        raise ValueError("examples and max_images must be positive")
    model, run = load_checkpoint(checkpoint, device)
    transform = eval_transform(run["image_size"])
    names = class_names(split)
    split = split.select_columns(["image", "label"])
    output = Path(out_dir) / Path(checkpoint).stem
    output.mkdir(parents=True, exist_ok=True)
    counts = {"correct": 0, "wrong": 0}
    records = []
    scanned = 0
    explain = GradCAM(model)
    for index in range(min(len(split), max_images or len(split))):
        row = split[index]
        image = transform(row["image"]).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = model(image).softmax(1)[0]
        predicted = int(probabilities.argmax().item())
        label = int(row["label"])
        outcome = "correct" if predicted == label else "wrong"
        scanned += 1
        if counts[outcome] >= examples:
            continue
        cam, _, target = explain(image, predicted)
        rgb = image[0].detach().cpu() * torch.tensor(IMAGENET_STD)[:, None, None]
        rgb = (rgb + torch.tensor(IMAGENET_MEAN)[:, None, None]).clamp(0, 1).permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis in axes:
            axis.imshow(rgb)
            axis.axis("off")
        axes[0].set_title("Model input (evaluation crop)")
        axes[1].imshow(cam.cpu().numpy(), cmap="jet", vmin=0, vmax=1, alpha=0.45)
        axes[1].set_title("Grad-CAM: predicted class")
        fig.suptitle(f"{run['name']} | {outcome}\nTrue: {names[label]} | Predicted: {names[predicted]} "
                     f"({float(probabilities[predicted]):.1%})", fontsize=10)
        filename = f"{outcome}_{index:05d}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        records.append({"index": index, "true_label": label, "true_species": names[label],
                        "predicted_label": predicted, "predicted_species": names[predicted],
                        "confidence": float(probabilities[predicted]), "target_class": target,
                        "outcome": outcome, "figure": filename})
        counts[outcome] += 1
        if all(count >= examples for count in counts.values()):
            break
    report = {"checkpoint": str(checkpoint), "run": run, "split": split_name,
              "scanned": scanned, "requested_per_category": examples, "selected": counts,
              "examples": records}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{run['name']}: scanned {scanned}; selected {counts}; saved to {output}")
    for category, count in counts.items():
        if count < examples:
            print(f"Only {count}/{examples} {category} examples found in the scanned images.")
    return report


def run_reports(args):
    data = load_from_disk(args.data_dir)
    for checkpoint in args.checkpoints:
        generate_report(checkpoint, data[args.split], args.out_dir, args.examples,
                        args.max_images, args.device, args.split)
