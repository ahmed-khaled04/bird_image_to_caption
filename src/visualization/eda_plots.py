"""Matplotlib figures for the EDA report. Every function saves one PNG and closes it."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def plot_image_grid(images: list, titles: list, path: Path, suptitle: str, cols: int = 4) -> None:
    """Show PIL images in a grid, one title per image."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3), squeeze=False, layout="constrained")
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img.convert("RGB"))  # grayscale (L) images are shown as gray, not with a colour map
        ax.set_title(title, fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(suptitle)
    save_figure(fig, path)


def plot_sample_images(split, class_names: list, path: Path, n: int = 16, seed: int = 0) -> None:
    """Grid of random images with their species and size."""
    records = [split[int(i)] for i in np.random.default_rng(seed).choice(len(split), size=n, replace=False)]
    images = [r["image"] for r in records]
    titles = [f"{class_names[r['label']]}\n{img.width}x{img.height}" for r, img in zip(records, images)]
    plot_image_grid(images, titles, path, "Random training images (species, width x height)")


def plot_grayscale_images(splits, stats: pd.DataFrame, class_names: list, path: Path) -> None:
    """All images stored in grayscale (mode L): they have no colour target for colorization."""
    gray = stats[stats["mode"] == "L"]
    if gray.empty:
        print("no grayscale images to plot")
        return
    images = [splits[row.split][int(row.index)]["image"] for row in gray.itertuples()]
    titles = [f"{row.split}: {class_names[row.label]}" for row in gray.itertuples()]
    plot_image_grid(images, titles, path, "Images stored in grayscale (mode L)")


def plot_image_sizes(stats: pd.DataFrame, path: Path, target_size: int = 224) -> None:
    """Width vs height, aspect ratio, and shorter side compared with the model input size."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")

    sns.scatterplot(data=stats, x="width", y="height", hue="split", s=8, alpha=0.4, ax=axes[0])
    axes[0].set_title("Image width vs height (px)")

    sns.histplot(data=stats, x="aspect_ratio", hue="split", bins=50, element="step", ax=axes[1])
    axes[1].axvline(1.0, ls="--", c="k", lw=0.8)
    axes[1].set_title("Aspect ratio (width / height)")

    shorter = stats[["width", "height"]].min(axis=1).rename("shorter side (px)")
    sns.histplot(x=shorter, hue=stats["split"], bins=50, element="step", ax=axes[2])
    axes[2].axvline(target_size, ls="--", c="r", lw=1)
    axes[2].text(target_size, axes[2].get_ylim()[1] * 0.95, f" {target_size}px", color="r", va="top")
    axes[2].set_title("Shorter side vs model input size")

    save_figure(fig, path)


def plot_image_modes(stats: pd.DataFrame, path: Path) -> None:
    """How many images use each colour mode (RGB, L = grayscale, CMYK, ...)."""
    counts = pd.crosstab(stats["mode"].fillna("unreadable"), stats["split"])
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    counts.plot.bar(ax=ax, rot=0)
    ax.set_yscale("log")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    ax.set_title("Colour modes per split (log scale)")
    ax.set_xlabel("PIL mode")
    ax.set_ylabel("images")
    save_figure(fig, path)


def plot_duplicate_pairs(splits, pairs: pd.DataFrame, path: Path) -> None:
    """Show every duplicate pair side by side so the detection can be checked by eye."""
    if pairs.empty:
        print("no duplicate pairs to plot")
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(7, 2.8 * len(pairs)), squeeze=False, layout="constrained")
    for row, pair in zip(axes, pairs.itertuples()):
        for ax, split_name, index, file_name in (
            (row[0], pair.split_a, pair.index_a, pair.file_a),
            (row[1], pair.split_b, pair.index_b, pair.file_b),
        ):
            ax.imshow(splits[split_name][int(index)]["image"].convert("RGB"))
            ax.set_title(f"{split_name}: {file_name}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"{pair.kind}\npHash {pair.phash_distance}, dHash {pair.dhash_distance}", fontsize=8)
    fig.suptitle("Duplicate image pairs (same photo, possibly edited)")
    save_figure(fig, path)
