"""Fine-tune EfficientNet-B0 on the brain MRI dataset.

Run with:  python -m brainmri.train
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn

from . import config
from .data import make_loaders
from .model import build_model, get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over a loader. Trains when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    use_amp = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def set_backbone_trainable(model, trainable):
    for param in model.features.parameters():
        param.requires_grad = trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=2,
                        help="epochs training only the new head")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="epochs training the whole network")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_loader, val_loader, _ = make_loaders(args.batch_size, args.workers)
    print(f"Train images: {len(train_loader.dataset)}, "
          f"validation images: {len(val_loader.dataset)}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    history, best_acc = [], 0.0
    phases = [
        # (name, epochs, backbone trainable, learning rate)
        ("head", args.head_epochs, False, 1e-3),
        ("finetune", args.finetune_epochs, True, 3e-4),
    ]
    epoch_no = 0
    for phase, epochs, backbone_trainable, lr in phases:
        if epochs == 0:
            continue
        set_backbone_trainable(model, backbone_trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for _ in range(epochs):
            epoch_no += 1
            start = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            row = dict(epoch=epoch_no, phase=phase,
                       train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc,
                       seconds=round(time.time() - start, 1))
            history.append(row)
            marker = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"state_dict": model.state_dict(),
                            "classes": config.CLASSES,
                            "epoch": epoch_no,
                            "val_acc": val_acc}, config.CHECKPOINT)
                marker = "  <- saved"
            print(f"[{phase:8s}] epoch {epoch_no:2d}  "
                  f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}  "
                  f"({row['seconds']}s){marker}", flush=True)

    with open(config.REPORT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to {config.CHECKPOINT}")


if __name__ == "__main__":
    main()
