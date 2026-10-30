"""Figures comparing grid search and random search."""
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter

from src.visualization.eda_plots import save_figure

LOG_SCALE = {"lr", "weight_decay"}


def plot_best_so_far(results: pd.DataFrame, metric: str, path: Path) -> None:
    """Best score found so far against the trial number and against the time spent."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, runs in results.groupby("search", sort=False):
        best = runs[metric].cummax()
        axes[0].step(runs["trial"], best, where="post", marker="o", label=name)
        axes[1].step(runs["seconds"].cumsum() / 60, best, where="post", marker="o", label=name)
    axes[0].set(xlabel="trial", ylabel=f"best {metric} so far", title="Best score per trial")
    axes[1].set(xlabel="minutes", ylabel=f"best {metric} so far", title="Best score per minute")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, path)


def plot_score_by_param(results: pd.DataFrame, metric: str, param_names: list, path: Path) -> None:
    """Score of every trial against each hyperparameter: the grid tries few values, random search many."""
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 3.6), layout="constrained",
                             sharey=True, squeeze=False)
    for ax, param in zip(axes[0], param_names):
        for name, runs in results.groupby("search", sort=False):
            ax.scatter(runs[param], runs[metric], label=name, alpha=0.75)
        if param in LOG_SCALE:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())  # label only the powers of 10
        ax.set(xlabel=param, title=param)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(metric)
    axes[0][0].legend()
    save_figure(fig, path)
