"""PyTorch datasets over the Hugging Face splits."""
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """(image tensor, species label) pairs. The transform decides whether the image is augmented."""

    def __init__(self, split, transform):
        self.split = split.select_columns(["image", "label"])  # reading a row then skips the text columns
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        row = self.split[index]
        return self.transform(row["image"]), row["label"]
