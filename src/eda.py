"""EDA pipeline: computes statistics with src.data, plots with src.visualization, prints a report."""
from pathlib import Path

import pandas as pd

from src.data import caption_stats
from src.data.caption_cleaning import TYPO_FIXES, clean_description
from src.data.class_stats import class_counts, imbalance_summary
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


def run_class_eda(splits, fig_dir: Path, table_dir: Path) -> None:
    split_names = list(splits.keys())
    counts = class_counts(splits, class_names(splits["train"]))
    summary = imbalance_summary(counts, split_names)

    table_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(table_dir / "class_counts.csv")

    print_class_report(counts, summary, split_names)

    eda_plots.plot_class_distribution(counts, split_names, fig_dir / "classes_distribution.png")
    eda_plots.plot_smallest_classes(counts, split_names, fig_dir / "classes_smallest.png")
    eda_plots.plot_family_sizes(counts, fig_dir / "classes_families.png")


def print_class_report(counts: pd.DataFrame, summary: pd.DataFrame, split_names: list) -> None:
    print("\n=== images per class")
    print(summary.to_string())

    for split_name in split_names:
        histogram = counts[split_name].value_counts().sort_index()
        print(f"\n{split_name}: how many species have N images")
        print(", ".join(f"{n}: {k}" for n, k in histogram.items()))

    print(f"\n=== species with the fewest {split_names[-1]} images")
    print(counts.nsmallest(10, split_names[-1])[split_names].to_string())

    families = counts.groupby("family").size()
    print(f"\n=== {len(families)} families; largest: "
          + ", ".join(f"{name} {size}" for name, size in families.nlargest(5).items()))


def run_caption_eda(splits, fig_dir: Path, table_dir: Path) -> None:
    raw = {name: list(split.select_columns(["description"])["description"]) for name, split in splits.items()}
    cleaned = {name: [clean_description(d) for d in descriptions] for name, descriptions in raw.items()}
    lines = {name: caption_stats.split_lines(descriptions) for name, descriptions in raw.items()}

    # Analysis is done on train; test is only used for the cleaning counts and the <unk> rate.
    train_lines, train_clean = lines["train"], [c for captions in cleaned["train"] for c in captions]
    per_image = caption_stats.per_image_counts(raw["train"], cleaned["train"])
    words_before = caption_stats.word_counts(train_lines)
    words_after = caption_stats.word_counts(train_clean)
    test_words = caption_stats.word_counts(c for captions in cleaned["test"] for c in captions)
    length_before = pd.Series([len(line.split()) for line in train_lines if line.strip()])
    length_after = pd.Series([len(caption.split()) for caption in train_clean])
    unknown_before = caption_stats.unknown_words(words_before)
    unknown_after = caption_stats.unknown_words(words_after)
    coverage = caption_stats.vocabulary_coverage(words_after, test_words)

    table_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(table_dir / "caption_vocab_coverage.csv", index=False)
    pd.DataFrame(unknown_after, columns=["word", "count"]).to_csv(
        table_dir / "caption_unknown_words_after_cleaning.csv", index=False)

    print_caption_report(raw, lines, cleaned, per_image, words_before, words_after,
                         unknown_before, unknown_after, length_before, length_after, coverage)

    eda_plots.plot_captions_per_image(per_image, fig_dir / "captions_per_image_before_after.png")
    eda_plots.plot_duplicates_per_image(per_image, fig_dir / "duplicates_per_image.png")
    eda_plots.plot_cleaning_funnel(caption_stats.cleaning_funnel(train_lines, cleaned["train"]),
                                   fig_dir / "cleaning_funnel.png")
    eda_plots.plot_formatting(caption_stats.formatting_issues(train_lines), caption_stats.symbol_counts(train_lines),
                              fig_dir / "formatting_and_punctuation.png")
    eda_plots.plot_caption_lengths(length_before, length_after, fig_dir / "caption_length_before_after.png")
    eda_plots.plot_typos(unknown_before, caption_stats.fixed_typos(words_before), TYPO_FIXES, fig_dir / "typos.png")
    eda_plots.plot_vocabulary(words_before, words_after, fig_dir / "vocabulary.png")
    eda_plots.plot_vocab_coverage(coverage, fig_dir / "captions_vocab_coverage.png")


def print_caption_report(raw, lines, cleaned, per_image, words_before, words_after,
                         unknown_before, unknown_after, length_before, length_after, coverage) -> None:
    print("\n=== cleaning: captions at each stage")
    for name in raw:
        funnel = caption_stats.cleaning_funnel(lines[name], cleaned[name])
        kept_per_image = pd.Series([len(c) for c in cleaned[name]]).value_counts().sort_index()
        print(f"{name}: {len(raw[name])} images, " + ", ".join(f"{k} {v}" for k, v in funnel.items()))
        print(f"   captions per image after cleaning: " + ", ".join(f"{n}: {k}" for n, k in kept_per_image.items()))

    print("\n=== raw train descriptions")
    print("lines per description:", per_image["lines"].value_counts().to_dict())
    print(f"images with an empty line: {(per_image['empty'] > 0).sum()}, "
          f"empty lines that are not the last line: {(per_image['empty'] > 1).sum()}")
    funnel = caption_stats.cleaning_funnel(lines["train"], cleaned["train"])
    print(f"images with duplicate captions: {(per_image['duplicates'] > 0).sum()} "
          f"({per_image['duplicates'].sum()} duplicate lines as written; cleaning drops "
          f"{funnel.iloc[2] - funnel['unique (kept)']}, because once punctuation "
          f"is removed more lines become identical)")

    print("\n=== formatting issues (raw train lines)")
    issues = caption_stats.formatting_issues(lines["train"])
    for name, count in issues.items():
        print(f"{name:25s} {count:6d}  ({count / len(lines['train']):.1%})")
    print("symbols:", caption_stats.symbol_counts(lines["train"]).most_common())

    print("\n=== caption length in words (train)")
    print(pd.DataFrame({"before": length_before.describe(), "after": length_after.describe()}).round(2).T)
    p95, p99 = length_after.quantile(0.95), length_after.quantile(0.99)
    print(f"after cleaning: p95 = {p95:.0f} words, p99 = {p99:.0f} words, "
          f"longer than p99: {(length_after > p99).sum()} captions")
    short = [c for captions in cleaned["train"] for c in captions if len(c.split()) <= 3]
    print(f"very short captions (<= 3 words): {len(short)}, e.g. {short[:8]}")

    print("\n=== vocabulary (train)")
    print(f"words before cleaning: {len(words_before)}, after: {len(words_after)}")
    print(f"words seen only once after cleaning: {sum(1 for c in words_after.values() if c == 1)}")
    print(f"unknown to the spell checker: {len(unknown_before)} -> {len(unknown_after)}")
    print(f"typo occurrences fixed: {sum(count for _, count in caption_stats.fixed_typos(words_before))}")
    print("most frequent remaining unknown words:", unknown_after[:20])

    print("\n=== vocabulary coverage (train words kept if seen >= min_count times)")
    shown = coverage[coverage["min_count"].isin([1, 2, 3, 5, 10, 20])].copy()
    shown["train_coverage"] = (shown["train_coverage"] * 100).round(2).astype(str) + "%"
    shown["test_unk_rate"] = (shown["test_unk_rate"] * 100).round(2).astype(str) + "%"
    print(shown.to_string(index=False))


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
