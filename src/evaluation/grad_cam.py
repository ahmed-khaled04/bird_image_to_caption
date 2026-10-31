"""Grad-CAM on the final spatial encoder features of a BirdClassifier."""
import torch
from torch.nn import functional as F


class GradCAM:
    """Explain one image using a predicted (default) or supplied class logit.

    Hooks are scoped to each call. Input gradients keep frozen encoders differentiable;
    autograd.grad avoids accumulating parameter gradients or changing frozen weights.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, image, target_class=None):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("GradCAM expects one image with shape (1, C, H, W)")
        modes = {module: module.training for module in self.model.modules()}
        features = []
        handle = self.model.encoder.register_forward_hook(
            lambda module, inputs, outputs: features.append(outputs[-1]))
        try:
            self.model.eval()
            with torch.enable_grad():
                logits = self.model(image.detach().clone().requires_grad_(True))
                target = int(logits.argmax(1).item()) if target_class is None else int(target_class)
                if not 0 <= target < logits.shape[1]:
                    raise ValueError("target_class is outside the model's class range")
                activation = features[-1]
                gradients, = torch.autograd.grad(logits[0, target], activation)
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activation).sum(dim=1, keepdim=True).relu()
                cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam[0, 0]
                cam = cam / cam.max().clamp_min(1e-12)
            return cam.detach(), logits.detach(), target
        finally:
            handle.remove()
            for module, training in modes.items():
                module.training = training
