"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Training Script
==========================================================
"""

import shutil
from ultralytics import YOLO

from config import (
    DATASET_PATH,
    MODEL_NAME,
    MODELS_DIR,
    OUTPUT_DIR,
    IMAGE_SIZE,
    EPOCHS,
    BATCH_SIZE,
    DEVICE,
    PROJECT_NAME,
    RUN_NAME,
)

from logger import logger
from utils import load_model, check_file_exists


def train_model():
    """
    Train the YOLO11 model.
    """

    logger.info("Checking dataset...")

    check_file_exists(DATASET_PATH)

    logger.info("Loading YOLO model...")

    model = YOLO(MODEL_NAME)

    logger.info("Training started...")

    results = model.train(
        data=str(DATASET_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR / PROJECT_NAME),
        name=RUN_NAME,
    )

    logger.info("Training completed successfully.")

    return results


def save_best_model():
    """
    Copy the trained best.pt model to models folder.
    """

    source = (
        OUTPUT_DIR
        / PROJECT_NAME
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    check_file_exists(source)

    destination = MODELS_DIR / "best.pt"

    shutil.copy(source, destination)

    logger.info(f"Best model saved to: {destination}")


def main():

    try:
        train_model()
        save_best_model()

    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()