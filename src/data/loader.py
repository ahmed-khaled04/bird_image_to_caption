from datasets import DatasetDict, load_dataset

TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"


def load_cub(cache_dir: str | None = None) -> DatasetDict:
    """Download (first run) or load from cache the CUB train/test splits."""
    train = load_dataset(TRAIN_REPO, split="train", cache_dir=cache_dir)
    test = load_dataset(TEST_REPO, split="test", cache_dir=cache_dir)
    return DatasetDict(train=train, test=test)


def class_names(split) -> list:
    """Species names if `label` is a ClassLabel, otherwise the unique label values."""
    feature = split.features["label"]
    return list(feature.names) if hasattr(feature, "names") else sorted(set(split["label"]))


def summarize(splits: DatasetDict) -> None:
    for name, split in splits.items():
        print(f"\n=== {name}: {split.num_rows} rows")
        print("features:", split.features)
        print("num classes:", len(class_names(split)))

    example = splits["train"][0]
    print("\n=== first train record")
    for key, value in example.items():
        if key == "image":
            print(f"image: mode={value.mode}, size={value.size}")
        elif key == "description":
            lines = value.split("\n")
            print(f"description: {len(lines)} lines")
            for line in lines:
                print("   ", repr(line))
        else:
            print(f"{key}: {value}")
