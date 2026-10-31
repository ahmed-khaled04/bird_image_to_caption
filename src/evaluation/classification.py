import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    top5_correct = 0
    seen = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        seen += batch_size

        top5 = logits.topk(k=5, dim=1).indices
        top5_correct += (
            (top5 == labels[:, None]).any(dim=1).sum().item()
        )

        all_labels.append(labels.cpu())
        all_predictions.append(logits.argmax(dim=1).cpu())

    if seen == 0:
        raise ValueError("Evaluation loader is empty")

    labels = torch.cat(all_labels).numpy()
    predictions = torch.cat(all_predictions).numpy()

    return {
        "loss": total_loss / seen,
        "accuracy": float((labels == predictions).mean()),
        "top5_accuracy": top5_correct / seen,
        "labels": labels,
        "predictions": predictions,
    }


def classification_tables(result, class_names):
    """Overall scores, one row per species, and a true-by-predicted confusion matrix."""
    labels, predictions = result["labels"], result["predictions"]
    class_ids = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, zero_division=0
    )
    per_class = pd.DataFrame({
        "species": class_names, "support": support,
        "precision": precision, "recall": recall, "f1": f1,
    })
    confusion = pd.DataFrame(
        confusion_matrix(labels, predictions, labels=class_ids),
        index=pd.Index(class_names, name="true_species"),
        columns=pd.Index(class_names, name="predicted_species"),
    )
    summary = {
        "images": len(labels), "loss": result["loss"],
        "accuracy": result["accuracy"], "top5_accuracy": result["top5_accuracy"],
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }
    return summary, per_class, confusion
