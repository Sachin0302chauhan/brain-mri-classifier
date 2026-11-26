"""Image transforms and data loaders."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config


def eval_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def train_transform():
    # MRI slices vary in orientation, brightness and framing, so the
    # augmentations imitate those differences rather than adding colour noise.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def _check_classes(dataset):
    if dataset.classes != config.CLASSES:
        raise RuntimeError(
            f"Expected class folders {config.CLASSES}, found {dataset.classes}"
        )


def make_loaders(batch_size=32, num_workers=4):
    """Split Training/ into train and validation; Testing/ stays untouched."""
    train_full = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform())
    val_full = datasets.ImageFolder(config.TRAIN_DIR, transform=eval_transform())
    test_set = datasets.ImageFolder(config.TEST_DIR, transform=eval_transform())
    _check_classes(train_full)
    _check_classes(test_set)

    indices = np.arange(len(train_full))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.VAL_FRACTION,
        stratify=train_full.targets,
        random_state=config.SEED,
    )

    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin, persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, **common)
    val_loader = DataLoader(Subset(val_full, val_idx), shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
