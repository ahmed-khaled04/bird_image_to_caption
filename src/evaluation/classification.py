import torch


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