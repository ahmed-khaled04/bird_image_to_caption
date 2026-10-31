"""Build models by name, so a config line can swap the encoder without touching the training code."""
from functools import partial

from torch import nn

from src.models.classifier import BirdClassifier
from src.models.encoder import ScratchEncoder
from src.models.pretrained import EfficientNetEncoder, ResNetEncoder

ENCODERS = {
    "scratch": ScratchEncoder,
    "resnet18": partial(ResNetEncoder, depth=18),
    "resnet50": partial(ResNetEncoder, depth=50),
    "efficientnet_b0": EfficientNetEncoder,
}


def build_encoder(name: str, **kwargs) -> nn.Module:
    """`kwargs` go to the encoder, e.g. freeze=True or weights=None for the pretrained ones."""
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder '{name}', choose from {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)


def build_classifier(encoder: str = "scratch", num_classes: int = 200, dropout: float = 0.3,
                     **encoder_kwargs) -> BirdClassifier:
    model = BirdClassifier(encoder=build_encoder(encoder, **encoder_kwargs), num_classes=num_classes, dropout=dropout)
    # Reconstruction metadata only: no changes to parameters or the forward pass.
    model.checkpoint_config = {
        "architecture": "BirdClassifier", "encoder": encoder, "num_classes": num_classes,
        "dropout": dropout, "freeze": bool(getattr(model.encoder, "frozen", False)),
        "encoder_kwargs": {key: value for key, value in encoder_kwargs.items()
                           if key not in {"weights", "freeze"}},
    }
    return model
