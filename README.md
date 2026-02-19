# 🪪 Egyptian ID Card Detection & Line Segmentation (YOLOv8)

This repository implements a **computer vision pipeline** for detecting **Egyptian ID cards** in images and **segmenting individual text lines** inside the detected ID card using **YOLOv8**.

The project is designed for **document analysis**, **OCR preprocessing**, and **identity document automation** tasks.

---

## 📌 Project Overview

The pipeline performs the following steps:

1. Detect the **Egyptian ID card** in an input image
2. Crop the detected ID card region
3. Detect **text lines** inside the ID card
4. Crop and save each detected line as a separate image

The system uses **two trained YOLOv8 models**, one for ID detection and one for line detection.

---

## 🧠 Models Used

| Model File | Description |
|-----------|------------|
| `ID_detection.pt` | Detects the ID card region |
| `line_detection.pt` | Detects text lines inside the ID card |
| `yolov8n.pt` | Base YOLOv8 architecture |

All models are loaded using the **Ultralytics YOLO API**.

---

## 🔄 Detection Pipeline

1. Load input image using OpenCV
2. Run **ID card detection**
3. Extract bounding box of detected ID card
4. Crop ID card from original image
5. Run **line detection** on the cropped ID card
6. Crop and save each detected line as an image

---

## 🛠️ Technologies Used

- Python 3.9+
- OpenCV
- Ultralytics YOLOv8
- PyTorch (backend for YOLO)

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install opencv-python ultralytics
