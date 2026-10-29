"""Class (species) distribution per split, for EDA."""
import pandas as pd


def class_counts(splits, class_names: list) -> pd.DataFrame:
    """Images per species: one row per class, one column per split, plus the bird family."""
    counts = pd.DataFrame({name: pd.Series(split["label"]).value_counts() for name, split in splits.items()})
    counts = counts.reindex(range(len(class_names)), fill_value=0).astype(int)
    counts.index = pd.Index(class_names, name="species")
    # Last word of the name groups related species: "Black footed Albatross" -> "Albatross".
    counts["family"] = [name.split()[-1] for name in class_names]
    return counts


def imbalance_summary(counts: pd.DataFrame, split_names: list) -> pd.DataFrame:
    """Per split: images per class (min/max/mean/std), max/min ratio and coefficient of variation."""
    summary = counts[split_names].agg(["min", "max", "mean", "std"]).T
    summary["max_min_ratio"] = summary["max"] / summary["min"]
    summary["cv"] = summary["std"] / summary["mean"]  # spread relative to the mean; 0 = perfectly balanced
    return summary.round(2)
