"""Train/validation splitting."""
import pandas as pd


def duplicate_train_files(pairs: pd.DataFrame) -> set:
    """Train images to drop before splitting, from the duplicate pairs found by `eda --part images`.

    - the train copy of a photo that is also in test: we would otherwise train on test images (leakage)
    - the second copy of a photo that is in train twice: it adds nothing, and the two copies could
      end up one in train and one in validation
    Test is never changed, so scores stay comparable with published CUB results.
    """
    cross = pairs[pairs["split_a"] != pairs["split_b"]]
    train_copies = cross["file_a"].where(cross["split_a"] == "train", cross["file_b"])
    within_train = pairs[(pairs["split_a"] == "train") & (pairs["split_b"] == "train")]
    return set(train_copies) | set(within_train["file_b"])


def split_train_val(train, val_fraction=0.15, seed=42):
    """Carve a validation set out of `train`, stratified by class. Returns (train, validation).

    `train` must be a Hugging Face Dataset whose "label" column is a ClassLabel.
    The real test set is never touched here.
    """
    parts = train.train_test_split(test_size=val_fraction, stratify_by_column="label", seed=seed)
    return parts["train"], parts["test"]
