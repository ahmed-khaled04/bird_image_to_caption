"""Cleaning pipeline: cleans the captions, builds the validation split, reports, plots and saves."""
from pathlib import Path

import pandas as pd

from src.data import caption_stats
from src.data.class_stats import class_counts
from src.data.loader import class_names
from src.data.prepare import prepare_splits
from src.data.splits import duplicate_train_files
from src.eda import run_caption_eda
from src.visualization import eda_plots


def run_cleaning(splits, val_fraction: float, seed: int, out_dir: Path, fig_dir: Path, table_dir: Path,
                 duplicates_path: Path, plots: bool = True) -> None:
    if not duplicates_path.exists():
        raise SystemExit(f"{duplicates_path} not found: run `python -m src.cli eda --part images` first")
    drop_files = duplicate_train_files(pd.read_csv(duplicates_path))
    print(f"\n=== dropping {len(drop_files)} duplicate train images before splitting (test is not changed)")
    print("\n".join(sorted(drop_files)))

    data = prepare_splits(splits["train"], splits["test"], val_fraction, seed, drop_files)
    split_names = list(data.keys())

    print_cleaning_report(data)
    problems = {name: caption_stats.caption_problems(data[name]["captions"]) for name in split_names}
    print_problems(problems)
    if any(count for split_problems in problems.values() for count in split_problems.values()):
        raise SystemExit("cleaning left problems in the captions, nothing was saved")

    counts = class_counts(data, class_names(data["train"]))
    print_split_report(data, counts)

    table_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(table_dir / "split_class_counts.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(out_dir))
    print(f"\nsaved the cleaned splits to {out_dir} (load with datasets.load_from_disk)")

    if plots:
        # the same caption figures as the notebook, computed on the full train set before it is split
        run_caption_eda(splits, fig_dir, table_dir)
        eda_plots.plot_split_class_counts(counts, split_names, fig_dir / "split_class_distribution.png")


def print_cleaning_report(data) -> None:
    print("\n=== cleaning: captions at each stage")
    rows = {}
    for name, split in data.items():
        raw, cleaned = list(split["description"]), list(split["captions"])
        funnel = caption_stats.cleaning_funnel(caption_stats.split_lines(raw), cleaned)
        per_image = pd.Series([len(captions) for captions in cleaned])
        long_enough = funnel.iloc[2]  # captions with at least MIN_WORDS words
        rows[name] = {"images": len(cleaned), **funnel.to_dict(),
                      "empty dropped": int(funnel["lines after split"] - funnel["non-empty"]),
                      "too short dropped": int(funnel["non-empty"] - long_enough),
                      "duplicates dropped": int(long_enough - funnel["unique (kept)"]),
                      "min per image": per_image.min(), "max per image": per_image.max()}
    print(pd.DataFrame(rows).to_string())


def print_problems(problems: dict) -> None:
    print("\n=== problems left after cleaning (all should be 0)")
    print(pd.DataFrame(problems).to_string())


def print_split_report(data, counts: pd.DataFrame) -> None:
    sizes = pd.Series({name: len(split) for name, split in data.items()})
    print("\n=== split sizes")
    print(sizes.to_string())
    if "validation" in sizes:
        print(f"validation share of the original train set: {sizes['validation'] / (sizes['train'] + sizes['validation']):.1%}")
        per_class = counts["validation"][counts["validation"] > 0]
        print(f"species in train: {int((counts['train'] > 0).sum())}, in validation: {len(per_class)}")
        print(f"validation images per species (min / max): {per_class.min()} / {per_class.max()}")
