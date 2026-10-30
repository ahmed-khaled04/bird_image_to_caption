"""CNN encoder trained from scratch.

An encoder maps an image to a list of feature maps, one per stage, from fine to coarse. It exposes
`out_channels` and `strides` so any head or decoder can be built on top of it without knowing which
encoder it is: the classifier uses the last map, a U-Net would use all of them as skip connections,
and a caption decoder would attend over the last map.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection. `stride` 2 halves the height and width."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        # the skip needs a 1x1 convolution when the shape changes
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ScratchEncoder(nn.Module):
    """Stem plus four residual stages: 224x224 input gives maps of 112, 56, 28, 14 and 7 pixels per side."""

    def __init__(self, widths: tuple = (32, 64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=widths[0]),
            nn.ReLU(inplace=True))
        self.stages = nn.ModuleList(
            ResidualBlock(in_channels=widths[i], out_channels=widths[i + 1], stride=2)
            for i in range(len(widths) - 1))
        self.out_channels = list(widths)
        self.strides = [2 ** (i + 1) for i in range(len(widths))]
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)  # the residual branch starts at zero, so each block begins as just its shortcut

    def forward(self, x: torch.Tensor) -> list:
        """Feature maps of every stage, finest first: [(B,32,112,112), (B,64,56,56), ..., (B,512,7,7)]."""
        features = [self.stem(x)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        return features
