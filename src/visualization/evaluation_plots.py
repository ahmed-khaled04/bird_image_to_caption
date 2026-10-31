"""Readable plots of the most frequent classification mistakes."""
import pandas as pd
from matplotlib import pyplot as plt

from src.visualization.eda_plots import save_figure


def plot_top_confusions(pairs: pd.DataFrame, path, title: str) -> None:
    shown = pairs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    if shown.empty:
        ax.text(0.5, 0.5, "No misclassifications", ha="center", va="center")
        ax.axis("off")
    else:
        labels = shown["true_species"] + " → " + shown["predicted_species"]
        ax.barh(labels, shown["count"])
        ax.set_xlabel("images")
        ax.grid(axis="x", alpha=0.3)
    ax.set_title(title)
    save_figure(fig, path)
