"""Evaluate the trained model on the held-out test set.

Run with:  python -m brainmri.evaluate
Writes reports/metrics.json, reports/confusion_matrix.png and
reports/training_curves.png.
"""

import json

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .data import make_loaders  # noqa: E402
from .model import get_device, load_trained  # noqa: E402

LABELS = [config.CLASS_LABELS[c] for c in config.CLASSES]


@torch.no_grad()
def predict_loader(model, loader, device):
    all_true, all_prob = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_prob.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_true.append(labels.numpy())
    return np.concatenate(all_true), np.concatenate(all_prob)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(path):
    history_file = config.REPORT_DIR / "history.json"
    if not history_file.exists():
        return
    history = json.loads(history_file.read_text())
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set_title("Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="validation")
    ax2.set_title("Accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    model = load_trained(device=device)
    _, _, test_loader = make_loaders(batch_size=64, num_workers=4)

    y_true, y_prob = predict_loader(model, test_loader, device)
    y_pred = y_prob.argmax(1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=LABELS,
                                   digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    config.REPORT_DIR.mkdir(exist_ok=True)
    metrics = {
        "test_images": int(len(y_true)),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            name: {k: round(float(v), 4) for k, v in report[name].items()}
            for name in LABELS
        },
        "confusion_matrix": cm.tolist(),
    }
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, config.REPORT_DIR / "confusion_matrix.png")
    plot_history(config.REPORT_DIR / "training_curves.png")

    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))
    print(f"Test accuracy: {accuracy:.4f}   macro F1: {macro_f1:.4f}")
    print(f"Reports written to {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
