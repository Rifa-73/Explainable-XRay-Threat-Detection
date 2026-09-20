"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Model Evaluation Script
==========================================================
"""

from config import (
    DATASET_PATH,
    MODELS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
)

from logger import logger
from utils import load_model, check_file_exists


def evaluate_model():
    """
    Evaluate the trained YOLO11 model.
    """

    model_path = MODELS_DIR / "best.pt"

    check_file_exists(model_path)
    check_file_exists(DATASET_PATH)

    model = load_model(model_path)

    logger.info("Starting model evaluation...")

    metrics = model.val(
        data=str(DATASET_PATH),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
    )

    logger.info("Evaluation completed successfully.")

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    print(f"mAP@50        : {metrics.box.map50:.4f}")
    print(f"mAP@50-95     : {metrics.box.map:.4f}")
    print("=" * 50)


def main():

    try:
        evaluate_model()

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()