"""ImageNet-pretrained encoders from torchvision, wrapped to the same interface as ScratchEncoder.

Each wrapper cuts the torchvision network into five stages and returns their outputs, finest first, at
strides 2, 4, 8, 16 and 32 (112, 56, 28, 14 and 7 pixels per side for a 224x224 input). The ImageNet
classifier on top is dropped; BirdClassifier puts its own 200-class layer on the last map.

The inputs must be normalized with the ImageNet mean and std, which src.data.transforms already does.
"""
import torch
from torch import nn
from torchvision import models

STRIDES = [2, 4, 8, 16, 32]


class PretrainedEncoder(nn.Module):
    """Runs `stages` one after the other and returns every stage's output.

    With `freeze=True` the encoder is a fixed feature extractor: its weights get no gradient and it always
    stays in eval mode, so BatchNorm keeps the ImageNet running statistics instead of drifting to the
    statistics of our batches (requires_grad=False alone does not stop that).
    """

    def __init__(self, stages: list, out_channels: list, freeze: bool = False):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.out_channels = out_channels
        self.strides = STRIDES
        self.frozen = freeze
        if freeze:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode: bool = True):
        return super().train(mode and not self.frozen)

    def forward(self, x: torch.Tensor) -> list:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


RESNETS = {
    18: (models.resnet18, models.ResNet18_Weights, [64, 64, 128, 256, 512]),
    50: (models.resnet50, models.ResNet50_Weights, [64, 256, 512, 1024, 2048]),
}


class ResNetEncoder(PretrainedEncoder):
    """ResNet-18 or ResNet-50: the stem (stride 2), then max pooling plus layer1 (stride 4), layer2, layer3 and layer4.

    `weights` is a torchvision weights name such as "IMAGENET1K_V1", or None for random initialization.
    """

    def __init__(self, depth: int = 18, weights: str = "IMAGENET1K_V1", freeze: bool = False):
        build, weight_enum, channels = RESNETS[depth]
        net = build(weights=weight_enum[weights] if weights else None)
        stages = [nn.Sequential(net.conv1, net.bn1, net.relu),
                  nn.Sequential(net.maxpool, net.layer1),
                  net.layer2, net.layer3, net.layer4]
        super().__init__(stages, channels, freeze)


class EfficientNetEncoder(PretrainedEncoder):
    """EfficientNet-B0. torchvision splits it into nine blocks; blocks that share a resolution are grouped
    so each stage ends where the resolution halves: (0, 1) stride 2, (2) 4, (3) 8, (4, 5) 16, (6, 7, 8) 32.
    """

    def __init__(self, weights: str = "IMAGENET1K_V1", freeze: bool = False):
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights[weights] if weights else None)
        blocks = net.features
        stages = [blocks[0:2], blocks[2], blocks[3], blocks[4:6], blocks[6:9]]
        super().__init__(stages, [16, 24, 40, 112, 1280], freeze)
