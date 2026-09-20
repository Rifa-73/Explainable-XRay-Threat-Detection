"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Single Image Prediction Script
==========================================================
"""

from pathlib import Path
import cv2
import pandas as pd


from config import MODELS_DIR, OUTPUT_DIR, CONFIDENCE_THRESHOLD
from logger import logger
from utils import load_model, check_file_exists, create_directory


def show_help():
    """Display keyboard shortcuts."""

    print("\n" + "=" * 60)
    print("Keyboard Shortcuts")
    print("=" * 60)
    print(" Q   : Quit")
    print(" ESC : Quit")
    print(" S   : Save Snapshot")
    print(" I   : Show Image Information")
    print(" H   : Show Help")
    print("=" * 60)


def predict_image():

    model_path = MODELS_DIR / "best.pt"

    check_file_exists(model_path)

    model = load_model(model_path)

    image_path = Path(input("Enter image path: ").strip())

    check_file_exists(image_path)

    prediction_dir = OUTPUT_DIR / "predictions"

    create_directory(prediction_dir)

    logger.info("Running prediction...")

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False
    )

    result = results[0]

    
    output_image = result.plot()

    output_path = prediction_dir / image_path.name

    cv2.imwrite(str(output_path), output_image)

    logger.info(f"Prediction saved at: {output_path}")

    print(f"\nPrediction saved at:\n{output_path}")

    show_help()

    while True:

        cv2.imshow("Prediction", output_image)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q") or key == 27:

            logger.info("Prediction window closed.")

            break

        # Save Snapshot
        elif key == ord("s"):

            snapshot_path = prediction_dir / f"{image_path.stem}_snapshot.jpg"

            cv2.imwrite(str(snapshot_path), output_image)

            logger.info(f"Snapshot saved at: {snapshot_path}")

            print(f"\nSnapshot saved:\n{snapshot_path}")

        # Image Information
        elif key == ord("i"):

            print("\n" + "=" * 60)
            print("Image Information")
            print("=" * 60)
            print(f"File Name : {image_path.name}")
            print(f"Location  : {image_path}")
            print(f"Width     : {output_image.shape[1]} px")
            print(f"Height    : {output_image.shape[0]} px")
            print("=" * 60)

        # Help
        elif key == ord("h"):

            show_help()

    cv2.destroyAllWindows()


def main():

    try:
        predict_image()

    except Exception as e:

        logger.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()

