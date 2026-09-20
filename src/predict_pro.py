"""
==========================================================
Project:
Explainable X-Ray Threat Detection using YOLO11

Professional Interactive Prediction Tool
Version : 2.0
==========================================================
"""

from pathlib import Path
from collections import Counter

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    MODELS_DIR,
    OUTPUT_DIR,
    CONFIDENCE_THRESHOLD,
)

from logger import logger
from utils import (
    load_model,
    check_file_exists,
    create_directory,
)


# ----------------------------------------------------------
# Keyboard Help
# ----------------------------------------------------------

def show_help():

    print("\n" + "=" * 65)
    print("              Keyboard Shortcuts")
    print("=" * 65)
    print(" Q   : Quit")
    print(" ESC : Quit")
    print(" S   : Save Snapshot")
    print(" L   : Save Detection CSV")
    print(" T   : Detection Table")
    print(" C   : Count Objects")
    print(" G   : Generate Detection Graph")
    print(" I   : Image Information")
    print(" H   : Show Help")
    print("=" * 65)


# ----------------------------------------------------------
# Detection Summary
# ----------------------------------------------------------

def get_detections(result, model):

    detections = []

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        detections.append(
            {
                "Class": class_name,
                "Confidence": round(confidence, 4)
            }
        )

    return detections


# ----------------------------------------------------------
# Print Detection Summary
# ----------------------------------------------------------

def print_detection_summary(detections):

    print("\n" + "=" * 65)
    print("Detection Summary")
    print("=" * 65)

    if len(detections) == 0:

        print("No Threat Objects Detected.")

    else:

        for index, detection in enumerate(detections, start=1):

            print(f"{index}. {detection['Class']}")
            print(f"   Confidence : {detection['Confidence']:.4f}")

        print("\nTotal Objects :", len(detections))

    print("=" * 65)


# ----------------------------------------------------------
# Save CSV
# ----------------------------------------------------------

def save_csv(detections, csv_path):

    if len(detections) == 0:

        print("No detections available.")

        return

    df = pd.DataFrame(detections)

    df.to_csv(csv_path, index=False)

    logger.info(f"CSV saved : {csv_path}")

    print(f"\nCSV Saved:\n{csv_path}")


# ----------------------------------------------------------
# Detection Table
# ----------------------------------------------------------

def show_table(detections):

    if len(detections) == 0:

        print("No detections available.")

        return

    df = pd.DataFrame(detections)

    print("\n")
    print(df.to_string(index=False))


# ----------------------------------------------------------
# Object Count
# ----------------------------------------------------------

def count_objects(detections):

    if len(detections) == 0:

        print("No detections available.")

        return

    counter = Counter()

    for detection in detections:

        counter[detection["Class"]] += 1

    print("\nObject Counts")

    for key, value in counter.items():

        print(f"{key:<15} : {value}")


# ----------------------------------------------------------
# Prediction Function
# ----------------------------------------------------------

def predict_image():

    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------

    model_path = MODELS_DIR / "best.pt"

    check_file_exists(model_path)

    model = load_model(model_path)

    image_path = Path(input("Enter image path: ").strip())

    check_file_exists(image_path)

    prediction_dir = OUTPUT_DIR / "predictions"

    create_directory(prediction_dir)

    logger.info("Running prediction...")

    # ------------------------------------------------------
    # Run Prediction
    # ------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False
    )

    result = results[0]

    detections = get_detections(result, model)

    print_detection_summary(detections)

    # ------------------------------------------------------
    # Draw Prediction
    # ------------------------------------------------------

    output_image = result.plot()

    output_path = prediction_dir / image_path.name

    cv2.imwrite(str(output_path), output_image)

    logger.info(f"Prediction saved at: {output_path}")

    print(f"\nPrediction saved at:\n{output_path}")

    show_help()

    # ------------------------------------------------------
    # Interactive Window
    # ------------------------------------------------------

    while True:

        cv2.imshow("Professional YOLO11 Prediction", output_image)

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------
        # Quit
        # ----------------------------------------------

        if key == ord("q") or key == 27:

            logger.info("Prediction window closed.")

            break

        # ----------------------------------------------
        # Save Snapshot
        # ----------------------------------------------

        elif key == ord("s"):

            snapshot_path = (
                prediction_dir /
                f"{image_path.stem}_snapshot.jpg"
            )

            cv2.imwrite(str(snapshot_path), output_image)

            logger.info(f"Snapshot saved at: {snapshot_path}")

            print(f"\nSnapshot saved:\n{snapshot_path}")

        # ----------------------------------------------
        # Save CSV
        # ----------------------------------------------

        elif key == ord("l"):

            csv_path = (
                prediction_dir /
                f"{image_path.stem}_detections.csv"
            )

            save_csv(detections, csv_path)

        # ----------------------------------------------
        # Detection Table
        # ----------------------------------------------

        elif key == ord("t"):

            show_table(detections)

        # ----------------------------------------------
        # Object Count
        # ----------------------------------------------

        elif key == ord("c"):

            count_objects(detections)


        # ----------------------------------------------
        # Generate Graph
        # ----------------------------------------------

        elif key == ord("g"):

            graph_path = (
                prediction_dir /
                f"{image_path.stem}_graph.png"
            )

            generate_graph(
                detections,
                graph_path
            )

        # ----------------------------------------------
        # Image Information
        # ----------------------------------------------

        elif key == ord("i"):

            print("\n" + "=" * 65)
            print("Image Information")
            print("=" * 65)
            print(f"File Name : {image_path.name}")
            print(f"Location  : {image_path}")
            print(f"Width     : {output_image.shape[1]} px")
            print(f"Height    : {output_image.shape[0]} px")
            print("=" * 65)

        # ----------------------------------------------
        # Help
        # ----------------------------------------------

        elif key == ord("h"):

            show_help()

    cv2.destroyAllWindows()

# ----------------------------------------------------------
# Generate Detection Graph
# ----------------------------------------------------------

def generate_graph(detections, graph_path):

    if len(detections) == 0:

        print("No detections available.")

        return

    counter = Counter()

    for detection in detections:

        counter[detection["Class"]] += 1

    classes = list(counter.keys())

    counts = list(counter.values())

    plt.figure(figsize=(8, 5))

    plt.bar(classes, counts)

    plt.title("Detected Objects")

    plt.xlabel("Object Class")

    plt.ylabel("Count")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(graph_path)

    plt.close()

    logger.info(f"Graph saved : {graph_path}")

    print(f"\nGraph Saved:\n{graph_path}")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    try:

        predict_image()

    except Exception as error:

        logger.error(f"Prediction failed : {error}")

        print(f"\nError : {error}")


if __name__ == "__main__":

    main()