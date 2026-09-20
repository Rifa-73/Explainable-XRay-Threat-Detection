"""
==========================================================
Utility Functions
==========================================================
"""

from pathlib import Path
from ultralytics import YOLO


def create_directory(directory: Path) -> None:
    """
    Create a directory if it does not exist.
    """
    directory.mkdir(parents=True, exist_ok=True)


def check_file_exists(file_path: Path) -> None:
    """
    Raise an error if the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")


def load_model(model_path: Path) -> YOLO:
    """
    Load a YOLO model.
    """
    check_file_exists(model_path)
    return YOLO(str(model_path))


def get_image_files(folder_path: Path):
    """
    Return all supported image files from a folder.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    return [
        file
        for file in folder_path.iterdir()
        if file.suffix.lower() in extensions
    ]