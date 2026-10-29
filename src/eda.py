"""EDA pipeline: computes statistics with src.data, plots with src.visualization, prints a report."""
from pathlib import Path

import pandas as pd

from src.data.image_stats import collect_image_stats, near_duplicate_pairs
from src.data.loader import class_names
from src.visualization import eda_plots


def run_image_eda(splits, fig_dir: Path, table_dir: Path) -> None:
    stats = pd.concat([collect_image_stats(split, name) for name, split in splits.items()], ignore_index=True)
    pairs = near_duplicate_pairs(stats)

    table_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(table_dir / "image_stats.csv", index=False)
    pairs.to_csv(table_dir / "duplicate_pairs.csv", index=False)

    print_image_report(stats, pairs)

    names = class_names(splits["train"])
    eda_plots.plot_sample_images(splits["train"], names, fig_dir / "images_samples.png")
    eda_plots.plot_image_sizes(stats, fig_dir / "images_sizes.png")
    eda_plots.plot_image_modes(stats, fig_dir / "images_modes.png")
    eda_plots.plot_grayscale_images(splits, stats, names, fig_dir / "images_grayscale.png")
    eda_plots.plot_duplicate_pairs(splits, pairs, fig_dir / "images_duplicates.png")


def print_image_report(stats: pd.DataFrame, pairs: pd.DataFrame) -> None:
    print("\n=== colour modes")
    print(pd.crosstab(stats["mode"].fillna("unreadable"), stats["split"]))

    print("\n=== file formats")
    print(pd.crosstab(stats["format"].fillna("unreadable"), stats["split"]))

    corrupted = stats[stats["corrupted"]]
    print(f"\n=== corrupted files: {len(corrupted)}")
    if not corrupted.empty:
        print(corrupted[["split", "file_name", "error"]].to_string(index=False))

    print("\n=== sizes")
    print(stats.groupby("split")[["width", "height", "aspect_ratio", "file_kb"]].describe().T.round(2))
    shorter = stats[["width", "height"]].min(axis=1)
    print(f"shorter side < 224 px: {(shorter < 224).sum()} images (they will be upscaled)")

    print("\n=== duplicate pairs (pHash <= 8 and dHash <= 10, out of 64 bits)")
    if pairs.empty:
        print("none")
    else:
        print(pairs.groupby(["kind", "same_label"]).size().rename("pairs").to_string())
        print(f"byte-identical files: {pairs['identical_file'].sum()}")
        columns = ["kind", "file_a", "file_b", "phash_distance", "dhash_distance"]
        print(pairs[columns].to_string(index=False))
