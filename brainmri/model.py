"""EfficientNet-B0 with a new 4-class head."""

import torch
from torch import nn
from torchvision import models

from . import config


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(config.CLASSES)),
    )
    return model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained(path=config.CHECKPOINT, device=None):
    device = device or get_device()
    model = build_model(pretrained=False)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model
