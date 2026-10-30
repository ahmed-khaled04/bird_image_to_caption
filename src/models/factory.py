"""Build models by name, so a config line can swap the encoder without touching the training code."""
from torch import nn

from src.models.classifier import BirdClassifier
from src.models.encoder import ScratchEncoder

ENCODERS = {"scratch": ScratchEncoder}


def build_encoder(name: str, **kwargs) -> nn.Module:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder '{name}', choose from {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)


def build_classifier(encoder: str = "scratch", num_classes: int = 200, dropout: float = 0.3) -> BirdClassifier:
    return BirdClassifier(encoder=build_encoder(encoder), num_classes=num_classes, dropout=dropout)
