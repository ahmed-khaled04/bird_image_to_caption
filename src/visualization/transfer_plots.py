"""Figures comparing the from-scratch baseline with the transfer learning runs."""
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.visualization.eda_plots import save_figure

LINE_STYLES = {"from scratch": ":", "frozen": "--", "fine-tune": "-"}


def plot_learning_curves(results: list, path: Path) -> None:
    """Validation accuracy and train/validation loss per epoch. One colour per encoder, one line style per mode."""
    encoders = list(dict.fromkeys(r["run"]["encoder"] for r in results))
    colours = {encoder: f"C{i}" for i, encoder in enumerate(encoders)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for r in results:
        history = pd.DataFrame(r["history"])
        style = {"color": colours[r["run"]["encoder"]], "linestyle": LINE_STYLES.get(r["mode"], "-")}
        axes[0].plot(history["epoch"], history["val_accuracy"], label=r["run"]["name"], **style)
        axes[1].plot(history["epoch"], history["val_loss"], **style)
        axes[1].plot(history["epoch"], history["train_loss"], alpha=0.35, **style)
    axes[0].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
    axes[1].set(xlabel="epoch", ylabel="loss", title="Loss (dark: validation, light: train)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.legend(loc="outside right center")
    save_figure(fig, path)


def plot_test_scores(table: pd.DataFrame, path: Path) -> None:
    """Test accuracy, top-5 accuracy and macro F1 of every run."""
    scores = table[["test_accuracy", "test_top5_accuracy", "test_f1_macro"]].dropna()
    scores.columns = ["accuracy", "top-5 accuracy", "macro F1"]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    scores.plot.bar(ax=ax, rot=30, width=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7)
    ax.set(ylabel="test score", ylim=(0, 1.05), title="Test scores (best validation epoch of each run)")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, path)
