"""Grad-CAM: highlights the image regions that drove a prediction.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        # The last convolutional block of EfficientNet-B0.
        self.target_layer = target_layer or model.features[-1]
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _module, _inputs, output):
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self._gradients = grad.detach()

    def __call__(self, image_tensor, class_index):
        """Return a heatmap in [0, 1] with the input's height and width."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            image_tensor = image_tensor.clone().requires_grad_(True)
            logits = self.model(image_tensor)
            logits[0, class_index].backward()

        # Weight each feature map by its average gradient, keep positive evidence.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay(image: Image.Image, heatmap: np.ndarray, alpha=0.45) -> Image.Image:
    """Blend a heatmap over the original image."""
    base = image.convert("RGB")
    heat = Image.fromarray(np.uint8(heatmap * 255)).resize(base.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, colored, alpha)
