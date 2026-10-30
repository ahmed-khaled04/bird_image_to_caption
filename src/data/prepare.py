"""Prepare already-loaded splits: clean the captions and build the validation split."""
from datasets import DatasetDict

from src.data.caption_cleaning import clean_descriptions
from src.data.splits import split_train_val


def prepare_splits(train, test, val_fraction=0.15, seed=42, drop_files=None):
    """Clean the `description` column of both datasets and split off a validation set.

    `train` and `test` are Hugging Face Datasets that are already loaded.
    `drop_files` are train file names removed before splitting (duplicates, see splits.duplicate_train_files).
    Returns a DatasetDict with "train", "validation" and "test"; each has a "captions" column
    (cleaned list of captions per image) next to the raw "description".
    With val_fraction=None or 0 there is no validation split, only "train" and "test".
    """
    if drop_files:
        train = train.filter(lambda file_name: file_name not in drop_files, input_columns="file_name")

    # clean before splitting so every split carries its captions
    train = train.add_column("captions", clean_descriptions(train["description"]))
    test = test.add_column("captions", clean_descriptions(test["description"]))

    if not val_fraction:
        return DatasetDict({"train": train, "test": test})

    train, val = split_train_val(train, val_fraction, seed)
    return DatasetDict({"train": train, "validation": val, "test": test})
