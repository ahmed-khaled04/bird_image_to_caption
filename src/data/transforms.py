"""Image transforms for the train split. Validation and test get no augmentation."""
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb(image):
    """8 images are stored as grayscale (mode L); the models expect three channels."""
    return image.convert("RGB")


def train_augmentation(image_size: int = 224) -> T.Compose:
    """Random changes a bird photo could plausibly have. PIL image in, PIL image out.

    - rotation: small tilt of the camera (before the crop, so the crop often cuts off the black corners)
    - random resized crop: different framing and distance; scale starts at 0.5, not torchvision's 0.08,
      because a small bird in the photo could be cropped out completely
    - horizontal flip: a bird facing left or right is the same species
    - colour jitter: lighting changes only; hue is almost untouched because plumage colour identifies the species
    - no vertical flip: birds are never photographed upside down
    """
    return T.Compose([
        T.Lambda(to_rgb),
        T.RandomRotation(15),
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    ])


def to_model_input() -> T.Compose:
    """PIL image to a normalized float tensor of shape (3, H, W)."""
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(image_size: int = 224) -> T.Compose:
    """Augmentation followed by tensor conversion: what the train loader applies to every image."""
    return T.Compose([train_augmentation(image_size), to_model_input()])
