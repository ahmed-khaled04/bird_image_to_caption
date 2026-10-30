"""Classification model: any encoder followed by global average pooling and a linear layer."""
import torch
from torch import nn


class BirdClassifier(nn.Module):
    """Encoder -> last feature map -> global average pooling -> dropout -> linear layer -> logits."""

    def __init__(self, encoder: nn.Module, num_classes: int = 200, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=encoder.out_channels[-1], out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = self.encoder(x)[-1]
        return self.fc(self.dropout(self.pool(last).flatten(start_dim=1)))
