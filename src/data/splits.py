"""Train/validation splitting."""


def split_train_val(train, val_fraction=0.15, seed=42):
    """Carve a validation set out of `train`, stratified by class. Returns (train, validation).

    `train` must be a Hugging Face Dataset whose "label" column is a ClassLabel.
    The real test set is never touched here.
    """
    parts = train.train_test_split(test_size=val_fraction, stratify_by_column="label", seed=seed)
    return parts["train"], parts["test"]
