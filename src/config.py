"""
==========================================================
Project:
Explainable Deep Learning Framework for Threat Detection
in X-Ray Baggage Images using YOLO11

Configuration File
==========================================================
"""

from pathlib import Path
import torch

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "X-Ray-Baggage-3" / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create folders automatically
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Training Configuration
# ==========================================================

MODEL_NAME = "yolo11n.pt"

IMAGE_SIZE = 640

EPOCHS = 50

BATCH_SIZE = 16

LEARNING_RATE = 0.001

CONFIDENCE_THRESHOLD = 0.15

# ==========================================================
# Device Configuration
# ==========================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ==========================================================
# Output Configuration
# ==========================================================

PROJECT_NAME = "Explainable_Xray_Project"

RUN_NAME = "YOLO11_Training"


# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = [
    "Gun",
    "Knife",
    "Pliers",
    "Scissors",
    "Wrench",
]