"""Checkpoint-based Grad-CAM reports, without training or downloading model weights."""
import base64
from html import escape
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


def write_gallery(reports, checkpoints, out_dir):
    """Create a portable HTML gallery containing the figures, with no external scripts."""
    sections = []
    for report, checkpoint in zip(reports, checkpoints):
        cards = []
        for record in report["examples"]:
            path = Path(out_dir) / Path(checkpoint).stem / record["figure"]
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            caption = escape(f"{record['outcome']} | dataset index {record['index']}")
            cards.append(f'<figure><img src="data:image/png;base64,{encoded}" '
                         f'alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        heading = escape(str(report["run"]["name"]))
        summary = escape(f"Split: {report['split']}; scanned: {report['scanned']}; selected: {report['selected']}")
        sections.append(f"<section><h2>{heading}</h2><p>{summary}</p>{''.join(cards)}</section>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Grad-CAM comparison</title><style>"
            "body{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
            "img{width:100%;height:auto}figure{margin:1rem 0}section{border-top:1px solid #aaa}"
            "</style><h1>Grad-CAM comparison</h1><p>Predicted-class explanations. Colours are "
            "normalized per image. Selected examples do not estimate overall accuracy.</p>"
            + ''.join(sections) + "</html>")
    path = Path(out_dir) / "comparison.html"
    path.write_text(page)
    print("Comparison gallery:", path)
    return path


def run_reports(args):
    data = load_from_disk(args.data_dir)
    if len({Path(path).stem for path in args.checkpoints}) != len(args.checkpoints):
        raise ValueError("Checkpoint filenames must have distinct stems to avoid overwriting reports")
    reports = [generate_report(checkpoint, data[args.split], args.out_dir, args.examples,
                               args.max_images, args.device, args.split)
               for checkpoint in args.checkpoints]
    write_gallery(reports, args.checkpoints, args.out_dir)
