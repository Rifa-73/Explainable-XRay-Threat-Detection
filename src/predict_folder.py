"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Batch Prediction Script
==========================================================
"""

from pathlib import Path

from config import MODELS_DIR, OUTPUT_DIR, CONFIDENCE_THRESHOLD
from logger import logger
from utils import (
    load_model,
    create_directory,
    get_image_files,
    check_file_exists,
)


def predict_folder():

    model_path = MODELS_DIR / "best.pt"

    check_file_exists(model_path)

    model = load_model(model_path)

    folder_path = Path(input("Enter folder path: ").strip())

    check_file_exists(folder_path)

    images = get_image_files(folder_path)

    if len(images) == 0:
        logger.warning("No images found in the selected folder.")
        return

    output_dir = OUTPUT_DIR / "batch_predictions"

    create_directory(output_dir)

    logger.info(f"Found {len(images)} images.")

    model.predict(
        source=str(folder_path),
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        project=str(output_dir),
        name="results",
        exist_ok=True,
        verbose=False,
    )

    logger.info("Batch prediction completed successfully.")
    logger.info(f"Results saved in: {output_dir}")


def main():

    try:
        predict_folder()

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")


if __name__ == "__main__":
    main()