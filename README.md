# 🔍 Explainable X-Ray Threat Detection using YOLO11

An Explainable AI-based computer vision framework for detecting potential threats in X-Ray baggage images using **YOLO11** and generating visual explanations using **Grad-CAM**.

The project is designed to combine automated object detection with model interpretability, helping users understand not only **what the model detects**, but also **which regions of the X-Ray image contributed to the prediction**.

---

## 📌 Project Overview

X-Ray baggage screening is an important component of security inspection systems. Traditional manual inspection of X-Ray images can be time-consuming and may be affected by human fatigue and visual complexity.

Deep learning-based object detection models can assist in identifying suspicious objects in X-Ray images. However, one major limitation of deep learning models is their lack of interpretability.

This project addresses this problem by combining:

- **YOLO11** for object detection
- **Grad-CAM** for visual explainability
- **OpenCV** for image processing
- **PyTorch** for deep learning
- Custom training and evaluation utilities
- Image, folder, video, and webcam inference

The framework produces both a detection result and an interpretable visual representation of the model's attention.

---

# 🎯 Objectives

The major objectives of this project are:

1. Detect potential threats in X-Ray baggage images using YOLO11.
2. Localize detected objects using bounding boxes.
3. Generate visual explanations for model predictions using Grad-CAM.
4. Provide multiple inference modes for images, folders, videos, and webcam input.
5. Develop a modular and reusable computer vision pipeline.
6. Evaluate the trained detection model using appropriate evaluation metrics.
7. Improve the interpretability of AI-assisted X-Ray threat detection.

---

# ✨ Key Features

### 🚨 1. X-Ray Threat Detection

The system uses YOLO11 to identify and localize potential threat objects in X-Ray baggage images.

### 🧠 2. Explainable AI

Grad-CAM is used to generate heatmaps that highlight image regions contributing to the model's prediction.

### 🎯 3. Object Localization

Detected objects are represented using bounding boxes along with their predicted class and confidence score.

### 🖼️ 4. Image Prediction

The framework supports inference on individual X-Ray images.

### 📂 5. Folder-Based Prediction

Multiple images can be processed automatically using the folder prediction pipeline.

### 🎥 6. Video Inference

The project includes functionality for processing video input and generating detection results.

### 📹 7. Webcam Inference

Real-time inference can be performed using a webcam where supported by the environment.

### 📊 8. Evaluation

The project includes evaluation utilities for analyzing model performance.

### ⚙️ 9. Configurable Pipeline

Training and inference parameters are maintained through configuration files, making the framework easier to modify and reproduce.

---

# 🧩 System Architecture

The overall pipeline can be represented as:

```text
                    ┌──────────────────────┐
                    │  X-Ray Baggage Image │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       YOLO11         │
                    │   Object Detection   │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Detection Result │   │ Target Detection│
          │ Class + BBox +   │   │   Selection      │
          │ Confidence       │   └────────┬─────────┘
          └──────────────────┘            │
                                          ▼
                                 ┌──────────────────┐
                                 │     Grad-CAM      │
                                 │ Explainability    │
                                 └────────┬─────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │   Heatmap /      │
                                 │ Visual Explanation│
                                 └────────┬─────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │ Explainable X-Ray│
                                 │ Detection Result │
                                 └──────────────────┘

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **PyTorch** | Deep learning framework |
| **Ultralytics YOLO11** | Object detection |
| **OpenCV** | Image and video processing |
| **NumPy** | Numerical computation |
| **Pandas** | Data processing |
| **Matplotlib** | Visualization |
| **Grad-CAM** | Explainable AI |
| **Git & GitHub** | Version control |

---