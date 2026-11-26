"""Classify a single MRI image from the command line.

Run with:  python -m brainmri.predict path/to/scan.jpg
"""

import argparse
from dataclasses import dataclass

import torch
from PIL import Image

from . import config
from .data import eval_transform
from .model import get_device, load_trained


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict  # display label -> probability
    class_index: int


class Predictor:
    def __init__(self, checkpoint=config.CHECKPOINT):
        self.device = get_device()
        self.model = load_trained(checkpoint, self.device)
        self.transform = eval_transform()

    def to_tensor(self, image: Image.Image):
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        logits = self.model(self.to_tensor(image))
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
        index = max(range(len(probs)), key=probs.__getitem__)
        labels = [config.CLASS_LABELS[c] for c in config.CLASSES]
        return Prediction(
            label=labels[index],
            confidence=probs[index],
            probabilities=dict(zip(labels, probs)),
            class_index=index,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    predictor = Predictor()
    for path in args.images:
        result = predictor.predict(Image.open(path))
        print(f"{path}: {result.label} ({result.confidence:.1%})")


if __name__ == "__main__":
    main()
