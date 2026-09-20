"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Real-Time Webcam Detection
==========================================================
"""

import cv2

from config import MODELS_DIR, CONFIDENCE_THRESHOLD, DEVICE
from logger import logger
from utils import load_model, check_file_exists


def start_webcam():

    model_path = MODELS_DIR / "best.pt"

    check_file_exists(model_path)

    model = load_model(model_path)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    logger.info("Webcam started.")
    logger.info("Press 'q' to quit.")

    while True:

        success, frame = cap.read()

        frame = cv2.flip(frame, 1)

        if not success:
            logger.error("Failed to capture frame.")
            break

        results = model.predict(
            source=frame,
            conf=0.15,
            imgsz=960,
            device=DEVICE,

            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO11 Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    logger.info("Webcam closed successfully.")


def main():

    try:
        start_webcam()

    except Exception as e:
        logger.error(f"Webcam failed: {e}")


if __name__ == "__main__":
    main()