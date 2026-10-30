"""Show training images before and after the train augmentation."""
from pathlib import Path

import numpy as np
import torch

from src.data.loader import class_names
from src.data.transforms import train_augmentation
from src.visualization.eda_plots import plot_image_grid


def run_augmentation_demo(splits, fig_dir: Path, indices: list | None, n_images: int, n_augmented: int,
                          image_size: int, seed: int) -> None:
    """One row per train image: the original, then `n_augmented` random augmentations of it."""
    train = splits["train"]
    names = class_names(train)
    if not indices:
        indices = [int(i) for i in np.random.default_rng(seed).choice(len(train), size=n_images, replace=False)]

    torch.manual_seed(seed)  # torchvision draws its random parameters from the torch generator
    augment = train_augmentation(image_size)

    images, titles = [], []
    for index in indices:
        record = train[index]
        original = record["image"]
        images.append(original)
        titles.append(f"{names[record['label']]} (train #{index})\noriginal {original.width}x{original.height}")
        for k in range(n_augmented):
            images.append(augment(original))
            titles.append(f"augmented {k + 1} ({image_size}x{image_size})")
        print(f"train #{index}: {names[record['label']]}, {record['file_name']}")

    plot_image_grid(images, titles, fig_dir / "augmentation_examples.png",
                    "Train augmentation: rotation, random resized crop, horizontal flip, colour jitter",
                    cols=n_augmented + 1)
