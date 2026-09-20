"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Video Detection Script
==========================================================
"""

from pathlib import Path
from collections import Counter
import time

import cv2

from config import (
    MODELS_DIR,
    OUTPUT_DIR,
    CONFIDENCE_THRESHOLD
)

from logger import logger

from utils import (
    load_model,
    check_file_exists,
    create_directory
)


# ----------------------------------------------------------
# Keyboard Help
# ----------------------------------------------------------

def show_help():

    print("\n" + "=" * 60)
    print("Keyboard Shortcuts")
    print("=" * 60)
    print(" Q   : Quit Video")
    print(" H   : Show Help")
    print("=" * 60)


# ----------------------------------------------------------
# Print Video Summary
# ----------------------------------------------------------

def print_summary(counter, total_frames, fps, output_path):

    print("\n" + "=" * 60)
    print("Video Detection Summary")
    print("=" * 60)

    print(f"Frames Processed : {total_frames}")

    print(f"Average FPS      : {fps:.2f}")

    print("\nDetected Objects")

    if len(counter) == 0:

        print("No Objects Detected")

    else:

        for name, count in counter.items():

            print(f"{name:<15}: {count}")

    print("\nOutput Video")

    print(output_path)

    print("=" * 60)


# ----------------------------------------------------------
# Video Prediction
# ----------------------------------------------------------

def predict_video():

    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------

    model_path = MODELS_DIR / "best.pt"

    check_file_exists(model_path)

    model = load_model(model_path)

    # ------------------------------------------------------
    # Input Video
    # ------------------------------------------------------

    video_path = Path(
        input("Enter video path: ").strip()
    )

    check_file_exists(video_path)

    # ------------------------------------------------------
    # Output Directory
    # ------------------------------------------------------

    output_dir = OUTPUT_DIR / "videos"

    create_directory(output_dir)

    output_path = output_dir / (
        video_path.stem + "_detected.mp4"
    )

    logger.info("Opening video...")

    # ------------------------------------------------------
    # Open Video
    # ------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ------------------------------------------------------
    # Video Writer
    # ------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    logger.info("Video loaded successfully.")

    print(f"\nResolution : {width} x {height}")

    print(f"FPS        : {fps:.2f}")

    print(f"Frames     : {total_frames}")

    show_help()

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    counter = Counter()

    processed_frames = 0

    start_time = time.time()

    # ------------------------------------------------------
    # Process Video Frame-by-Frame
    # ------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        processed_frames += 1

        # --------------------------------------------------
        # Run YOLO Prediction
        # --------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot()

        # --------------------------------------------------
        # Count Detected Objects
        # --------------------------------------------------

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            for box in boxes:

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                counter[class_name] += 1

        # --------------------------------------------------
        # Calculate FPS
        # --------------------------------------------------

        elapsed_time = time.time() - start_time

        current_fps = (
            processed_frames / elapsed_time
            if elapsed_time > 0
            else 0
        )

        # --------------------------------------------------
        # Display Information
        # --------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"FPS : {current_fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Frame : {processed_frames}/{total_frames}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # --------------------------------------------------
        # Save Frame
        # --------------------------------------------------

        writer.write(annotated_frame)

        # --------------------------------------------------
        # Display Window
        # --------------------------------------------------

        cv2.imshow(
            "YOLO11 Video Detection",
            annotated_frame
        )

        # --------------------------------------------------
        # Keyboard Controls
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Quit

        if key == ord("q"):

            logger.info("Video detection stopped.")

            break

        # Help

        elif key == ord("h"):

            show_help()

        # --------------------------------------------------
        # Progress (Every 100 Frames)
        # --------------------------------------------------

        if processed_frames % 100 == 0:

            print(
                f"Processed "
                f"{processed_frames}/{total_frames} frames..."
            )

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # ------------------------------------------------------
    # Final Statistics
    # ------------------------------------------------------

    total_time = time.time() - start_time

    average_fps = (
        processed_frames / total_time
        if total_time > 0
        else 0
    )

    logger.info("Video processing completed.")

    logger.info(f"Output video saved at: {output_path}")

    print_summary(
        counter,
        processed_frames,
        average_fps,
        output_path
    )


# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def main():

    try:

        predict_video()

    except KeyboardInterrupt:

        print("\nProcess interrupted by user.")

        logger.warning(
            "Video processing interrupted by user."
        )

    except Exception as error:

        logger.error(
            f"Video processing failed: {error}"
        )

        print(f"\nError: {error}")


# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":

    main()