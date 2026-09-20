"""
==========================================================
Grad-CAM for YOLO11
Explainable X-Ray Threat Detection
==========================================================
"""

# ==========================================================
# Imports
# ==========================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from ultralytics import YOLO

from config import (
    DEVICE,
    MODELS_DIR,
    OUTPUT_DIR,
    IMAGE_SIZE,
    CLASS_NAMES
)

# ==========================================================
# Load Model
# ==========================================================

MODEL_PATH = MODELS_DIR / "best.pt"

yolo = YOLO(MODEL_PATH)

model = yolo.model.to(DEVICE)

model.eval()

# ==========================================================
# Output Directory
# ==========================================================

GRADCAM_DIR = OUTPUT_DIR / "gradcam"

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Global Variables
# ==========================================================

feature_maps = None

gradients = None

# ==========================================================
# Forward Hook
# ==========================================================

def forward_hook(module, inputs, output):

    global feature_maps

    if isinstance(output, (list, tuple)):
        feature_maps = output[0]
    else:
        feature_maps = output


# ==========================================================
# Backward Hook
# ==========================================================

def backward_hook(module, grad_input, grad_output):

    global gradients

    if isinstance(grad_output, (list, tuple)):
        gradients = grad_output[0]
    else:
        gradients = grad_output


# ==========================================================
# Target Layer
# ==========================================================

TARGET_LAYER = model.model[22]

forward_handle = TARGET_LAYER.register_forward_hook(
    forward_hook
)

backward_handle = TARGET_LAYER.register_full_backward_hook(
    backward_hook
)

# ==========================================================
# GradCAM Class
# ==========================================================

class GradCAM:

    def __init__(self):

        self.model = model

        self.device = DEVICE

        self.image_path = None

        self.original_image = None

        self.rgb_image = None

        self.input_tensor = None

        self.outputs = None

        self.predictions = None

        self.scores = None

        self.boxes = None

        self.class_id = None

        self.detection_id = None

        self.score = None

        self.heatmap = None

        self.overlay = None

    # ======================================================
    # Load Image
    # ======================================================

    def load_image(self, image_path):

        self.image_path = Path(image_path)

        self.original_image = cv2.imread(str(self.image_path))

        if self.original_image is None:
            raise FileNotFoundError(
                f"Cannot load image: {self.image_path}"
            )

        self.rgb_image = cv2.cvtColor(
            self.original_image,
            cv2.COLOR_BGR2RGB
        )


    # ======================================================
    # Preprocess Image
    # ======================================================

    def preprocess(self):

        image = cv2.resize(
            self.rgb_image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = image.astype(np.float32)

        image /= 255.0

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        self.input_tensor = torch.tensor(
            image,
            dtype=torch.float32,
            device=self.device
        )

        self.input_tensor.requires_grad_(True)


    # ======================================================
    # Forward Pass
    # ======================================================

    def forward(self):

        self.model.zero_grad()

        self.outputs = self.model(self.input_tensor)

        predictions = self.outputs

        print("forward pass completed")

    # ======================================================
    # Select Best Detection
    # ======================================================

    def select_detection(self):

        if isinstance(self.outputs, (list,tuple)):
            pred = self.outputs[0]

        else:
            pred  = self.outputs

        pred = pred.squeeze(0)

        print("Prediction shape:", pred.shape)
        print(pred)

        class_scores = pred[4:, :]

        score, index = torch.max(class_scores, dim=1)

        class_id = torch.argmax(score)

        detection_id = index[class_id]

        target = class_scores[class_id, detection_id]

        print("Class scores shape:", class_scores.shape)

        return target
    # ======================================================
    # Backward Pass
    # ======================================================

    def backward(self):

        target = self.select_detection()

        self.model.zero_grad()

        if self.input_tensor.grad is not None:
            self.input_tensor.grad.zero_()

        target.backward(retain_graph=True)

        print("Backward pass completed.")


    # ======================================================
    # Generate CAM
    # ======================================================

    def generate_heatmap(self):

        global feature_maps
        global gradients

        if feature_maps is None:
            raise RuntimeError(
                "Feature maps not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Gradients not captured."
            )

        grads = gradients[0]

        fmap = feature_maps[0]

        weights = torch.mean(
            grads,
            dim=(1, 2),
            keepdim=True
        )

        cam = torch.sum(
            weights * fmap,
            dim=0
        )

        cam = F.relu(cam)

        print("Feature map shape:", fmap.shape)
        print("Gradient shape:", grads.shape)
        print("CAM max:", cam.max().item())
        print("CAM min:", cam.min().item())

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        cam = cv2.resize(
            cam,
            (
                self.original_image.shape[1],
                self.original_image.shape[0]
            )
        )

        self.heatmap = cam


    # ======================================================
    # Convert Heatmap to Color
    # ======================================================

    def create_colormap(self):

        heatmap = np.uint8(
            255 * self.heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        self.heatmap = heatmap


    # ======================================================
    # Overlay Heatmap
    # ======================================================

    def overlay_heatmap(self):

        overlay = cv2.addWeighted(
            self.original_image,
            0.6,
            self.heatmap,
            0.4,
            0
        )

        self.overlay = overlay


    # ======================================================
    # Draw Prediction
    # ======================================================

    def draw_prediction(self):


        try:

            result = yolo.predict(
                source=self.original_image,
                device=DEVICE,
                verbose=False
            )[0]

            if len(result.boxes) == 0:
                return

            box = result.boxes[0]

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            label = (
                f"{CLASS_NAMES[class_id]} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                self.overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                self.overlay,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        except Exception as e:

            print(f"Bounding box drawing failed: {e}")


    # ======================================================
    # Save Image
    # ======================================================

    def save(self):

        output_path = (
            GRADCAM_DIR /
            f"{self.image_path.stem}_gradcam.jpg"
        )

        cv2.imwrite(
            str(output_path),
            self.overlay
        )

        print("=" * 60)
        print("Grad-CAM saved successfully")
        print(output_path)
        print("=" * 60)


    # ======================================================
    # Run Pipeline
    # ======================================================

    def run(self, image_path):

        self.load_image(image_path)

        self.preprocess()

        self.forward()

        self.backward()

        self.generate_heatmap()

        self.create_colormap()

        self.overlay_heatmap()

        self.draw_prediction()

        self.save()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    IMAGE_FOLDER = Path("X-Ray-Baggage-3/test/images")

    gradcam = GradCAM()

    images = list(IMAGE_FOLDER.glob("*.jpg"))

    # Generate Grad-CAM for first 10 images
    for image in images[:10]:
        print(f"Processing: {image.name}")
        gradcam.run(image)

    forward_handle.remove()
    backward_handle.remove()

    print("=" * 60)
    print("All Grad-CAM images generated successfully!")
    print("=" * 60)