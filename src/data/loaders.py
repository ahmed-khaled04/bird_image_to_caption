"""DataLoaders for the classification task."""
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ClassificationDataset
from src.data.transforms import eval_transform, train_transform


def build_dataloaders(data, image_size: int = 224, batch_size: int = 32, num_workers: int = 4,
                      seed: int = 42) -> dict:
    """One DataLoader per split in `data` (a DatasetDict with "train", and "validation" and/or "test").

    Train gets the augmentation and is shuffled. The other splits get the deterministic transform and keep
    their order. No split drops its last incomplete batch, so every image is used in every epoch.
    The seed always fixes the shuffling. It also fixes the random augmentation when num_workers > 0, because
    each worker is seeded from the loader's generator; with num_workers=0 the augmentation uses torch's global
    random state, so call torch.manual_seed(seed) too if you need identical images.
    """
    loaders = {}
    for name, split in data.items():
        is_train = name == "train"
        transform = train_transform(image_size) if is_train else eval_transform(image_size)
        loaders[name] = DataLoader(
            ClassificationDataset(split, transform),
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,  # keep workers alive between epochs
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders
