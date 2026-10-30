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


def plot_class_distribution(counts: pd.DataFrame, split_names: list, path: Path) -> None:
    """Images per species for each split, species sorted by their test count."""
    order = counts.sort_values(split_names[::-1]).index
    fig, axes = plt.subplots(len(split_names), 1, figsize=(14, 3 * len(split_names)), sharex=True,
                             sharey=True, layout="constrained")
    for ax, split_name, color in zip(axes, split_names, ["tab:blue", "tab:orange"]):
        values = counts.loc[order, split_name]
        ax.bar(range(len(order)), values, width=1.0, color=color)
        ax.axhline(values.mean(), ls="--", c="k", lw=0.8)
        ax.set_title(f"{split_name}: {values.min()} to {values.max()} images per species, mean {values.mean():.1f}")
        ax.set_ylabel("images")
    axes[-1].set_xlabel(f"{len(order)} species, sorted by {split_names[-1]} count")
    fig.suptitle("Class distribution")
    save_figure(fig, path)


def plot_smallest_classes(counts: pd.DataFrame, split_names: list, path: Path, n: int = 15) -> None:
    """The species with the fewest test images, with their train count for comparison."""
    smallest = counts.nsmallest(n, split_names[-1]).sort_values(split_names[-1], ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.35 * n + 1.5), layout="constrained")
    smallest[split_names].plot.barh(ax=ax, color=["tab:blue", "tab:orange"])
    for container in ax.containers:
        ax.bar_label(container, fontsize=7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))  # outside the bars, so no label is hidden
    ax.set_title(f"{n} species with the fewest {split_names[-1]} images")
    ax.set_xlabel("images")
    ax.set_ylabel("")
    save_figure(fig, path)


def plot_family_sizes(counts: pd.DataFrame, path: Path, top: int = 15) -> None:
    """How many species share a family name: many similar species make the task fine-grained."""
    sizes = counts.groupby("family").size().nlargest(top).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.35 * top + 1.5), layout="constrained")
    bars = ax.barh(sizes.index, sizes.values, color="tab:green")
    ax.bar_label(bars, fontsize=8)
    ax.set_title(f"Species per family (top {top} of {counts['family'].nunique()} families)")
    ax.set_xlabel("species")
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


def plot_captions_per_image(per_image: pd.DataFrame, path: Path) -> None:
    """Lines per description after splitting on newlines vs captions left after cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, column, title, color in (
        (axes[0], "lines", "Lines per image after split (before cleaning)", "tab:blue"),
        (axes[1], "kept", "Captions per image after cleaning", "tab:green"),
    ):
        counts = per_image[column].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color=color)
        ax.bar_label(bars, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("captions per image")
        ax.set_xticks(counts.index)
    axes[0].set_ylabel("images")
    save_figure(fig, path)


def plot_duplicates_per_image(per_image: pd.DataFrame, path: Path) -> None:
    """For images that repeat a caption: how many repeated lines they have."""
    duplicates = per_image.loc[per_image["duplicates"] > 0, "duplicates"]
    counts = duplicates.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars, fontsize=8)
    ax.set_xticks(counts.index)
    ax.set_xlabel("duplicate captions in an image")
    ax.set_ylabel("images")
    ax.set_title(f"Images with duplicate captions: {len(duplicates)}")
    save_figure(fig, path)


def plot_cleaning_funnel(funnel: pd.Series, path: Path) -> None:
    """Caption count after each cleaning stage."""
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    bars = ax.bar(funnel.index, funnel.values, color=["tab:gray", "tab:blue", "tab:green"])
    ax.bar_label(bars)
    ax.set_ylabel("captions")
    ax.set_title("Caption count at each cleaning stage (train)")
    save_figure(fig, path)


def plot_formatting(issues: pd.Series, symbols, path: Path) -> None:
    """Formatting problems per caption, and how often each punctuation mark or symbol appears."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    bars = axes[0].barh(issues.index, issues.values)
    axes[0].bar_label(bars, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Formatting issues (raw lines affected)")

    symbol, count = zip(*symbols.most_common())
    axes[1].bar([repr(s)[1:-1] for s in symbol], count)
    axes[1].set_yscale("log")
    axes[1].set_title("Punctuation / symbol frequency (log scale)")
    axes[1].set_xlabel("symbol")
    save_figure(fig, path)


def plot_caption_lengths(before: pd.Series, after: pd.Series, path: Path) -> None:
    """Words per caption before and after cleaning, with the 95th/99th percentiles of the cleaned length."""
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bins = range(0, 45)
    ax.hist(before, bins=bins, alpha=0.5, label="before cleaning")
    ax.hist(after, bins=bins, alpha=0.5, label="after cleaning")
    for q in (0.95, 0.99):
        value = after.quantile(q)
        ax.axvline(value, ls="--", c="k", lw=0.8)
        ax.text(value, ax.get_ylim()[1] * 0.9, f" p{int(q * 100)}={value:.0f}", rotation=90, va="top")
    ax.set_xlabel("words per caption")
    ax.set_ylabel("captions")
    ax.legend()
    ax.set_title("Caption length (non-empty lines)")
    save_figure(fig, path)


def plot_typos(unknown: list, fixed: list, typo_fixes: dict, path: Path, top: int = 15) -> None:
    """Most frequent words the spell checker does not know, and the most frequent typos we fix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    words, counts = zip(*unknown[:top])
    axes[0].barh(words, counts)
    axes[0].set_title("Most frequent unknown words before typo fixes")

    typos, counts = zip(*fixed[:top])
    axes[1].barh([f"{typo} -> {typo_fixes[typo]}" for typo in typos], counts, color="tab:green")
    axes[1].set_title("Most frequent typos we fixed")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("occurrences")
    save_figure(fig, path)


def plot_vocabulary(words_before, words_after, path: Path, top: int = 25) -> None:
    """Most frequent words after cleaning, and the word frequency by rank (Zipf curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    words, counts = zip(*words_after.most_common(top))
    axes[0].bar(words, counts)
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].set_title(f"{top} most frequent words after cleaning")

    for name, words_ in (("before", words_before), ("after", words_after)):
        frequencies = sorted(words_.values(), reverse=True)
        axes[1].loglog(range(1, len(frequencies) + 1), frequencies, label=f"{name} (vocab {len(frequencies)})")
    axes[1].set_xlabel("word rank")
    axes[1].set_ylabel("frequency")
    axes[1].legend()
    axes[1].set_title("Word frequency by rank")
    save_figure(fig, path)


def plot_vocab_coverage(coverage: pd.DataFrame, path: Path) -> None:
    """Vocabulary size and test <unk> rate when train words seen fewer than N times are dropped."""
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(coverage["min_count"], coverage["vocab_size"], marker="o", color="tab:blue")
    ax.set_xlabel("minimum count in train captions to keep a word")
    ax.set_ylabel("vocabulary size", color="tab:blue")
    ax.set_xticks(coverage["min_count"])

    unk = ax.twinx()
    unk.plot(coverage["min_count"], coverage["test_unk_rate"] * 100, marker="s", color="tab:red")
    unk.set_ylabel("test words that become <unk> (%)", color="tab:red")
    ax.set_title("Caption vocabulary: size vs coverage")
    save_figure(fig, path)


def plot_split_class_counts(counts: pd.DataFrame, split_names: list, path: Path) -> None:
    """For each split: how many species have N images. Shows the validation split keeps all species."""
    fig, axes = plt.subplots(1, len(split_names), figsize=(5 * len(split_names), 4), layout="constrained")
    for ax, split_name in zip(axes, split_names):
        histogram = counts[split_name].value_counts().sort_index()
        bars = ax.bar(histogram.index, histogram.values)
        ax.bar_label(bars, fontsize=8)
        ax.set_title(f"{split_name}: {int(counts[split_name].sum())} images, {int((counts[split_name] > 0).sum())} species")
        ax.set_xlabel("images per species")
        ax.set_ylabel("species")
    save_figure(fig, path)
