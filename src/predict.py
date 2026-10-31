"""Single-image classification using the checkpoint's evaluation preprocessing."""
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch

from src.data.transforms import eval_transform
from src.experiment import pick_device
from src.training.checkpoints import load_checkpoint
from src.training.checkpoint_refs import read_checkpoint, verify_class_labels


@torch.inference_mode()
def predict_image(checkpoint, image_path, top_k=5, device="cpu", labels_path=None):
    if top_k < 1:
        raise ValueError("top-k must be positive")
    device = pick_device() if device == "auto" else torch.device(device)
    payload = read_checkpoint(checkpoint)
    names = (json.loads(Path(labels_path).read_text()) if labels_path else payload.get("class_labels"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Checkpoint lacks class labels; supply --labels with a JSON list in label-index order")
    verify_class_labels(payload, names)
    model, run = load_checkpoint(checkpoint, device)
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        inputs = eval_transform(run.get("image_size", 224))(image).unsqueeze(0).to(device)
    probabilities = model(inputs).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(top_k, len(names)))
    return {
        "image": str(Path(image_path)), "checkpoint": str(Path(checkpoint)),
        "run": run["name"], "encoder": run["encoder"], "device": str(device),
        "image_size": run.get("image_size", 224),
        "predictions": [{"class_id": index, "species": names[index], "probability": score}
                        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())],
    }


def run_prediction(args):
    result = predict_image(args.checkpoint, args.image, args.top_k, args.device, args.labels)
    content = json.dumps(result, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content + "\n")
    print(content)
    return result
