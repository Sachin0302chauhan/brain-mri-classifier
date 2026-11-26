"""Shared settings for training, evaluation and the web app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CHECKPOINT = MODEL_DIR / "efficientnet_b0_brain_mri.pt"

# Folder names in the dataset, in the order the model outputs them.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

IMAGE_SIZE = 224
# ImageNet statistics, because the backbone was pretrained on ImageNet.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SEED = 42
VAL_FRACTION = 0.15
