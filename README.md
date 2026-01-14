<p align="center">
  <h1 align="center">SegMeow</h1>
  <p align="center">
    <strong>A cute YOLO segmentation trainer with Streamlit UI</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#demo">Demo</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#google-colab">Colab</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/YOLOv8-Supported-green?style=flat-square" alt="YOLOv8">
  <img src="https://img.shields.io/badge/YOLOv11-Supported-green?style=flat-square" alt="YOLOv11">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

---

## Overview

**SegMeow** is a Streamlit web application for training YOLOv8 and YOLOv11 segmentation models using Label Studio COCO annotations. Convert your annotations, train models, and run inference - all from a simple web interface.

---

## Features

| Feature | Description |
|---------|-------------|
| **COCO to YOLO Converter** | Automatically convert Label Studio COCO JSON to YOLO segmentation format |
| **Multi-Model Support** | Train with YOLOv8 or YOLOv11 segmentation models |
| **5 Model Sizes** | Choose from Nano, Small, Medium, Large, or XLarge |
| **Web UI** | Simple Streamlit interface - no coding required |
| **Real-time Inference** | Upload images and visualize polygon detections instantly |
| **Auto Class Detection** | Automatically extracts class names from COCO annotations |

---

## Demo

<!-- Add your screenshots here -->
```
Upload Screenshot → [screenshot_upload.png]
Training Screenshot → [screenshot_training.png]
Inference Screenshot → [screenshot_inference.png]
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/sohanurislamshuvo/SegMeow.git
cd SegMeow

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Usage

### Step 1: Upload Data
- Upload your **COCO JSON** file (exported from Label Studio)
- Upload your **training images**

### Step 2: Configure Dataset
- Class names are auto-detected from COCO JSON
- Click **"Generate data.yaml"**

### Step 3: Train Model
- Select **Model Version**: YOLOv8 or YOLOv11
- Select **Model Size**: Nano → XLarge
- Set **Epochs** and **Image Size**
- Click **"Start Training"**

### Step 4: Run Inference
- Upload any image
- View detected polygons with confidence scores

---

## Project Structure

```
SegMeow/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── utils/
│   ├── coco_to_yolo_seg.py   # COCO to YOLO converter
│   └── trainer.py            # Training wrapper
├── dataset/                  # Generated during use
│   ├── images/train/
│   └── labels/train/
└── runs/segment/             # Training outputs
```

---

## Model Options

### YOLOv8 Segmentation

| Model | Size | Speed | Accuracy |
|:------|:----:|:-----:|:--------:|
| `yolov8n-seg` | 3.4 MB | Fastest | Good |
| `yolov8s-seg` | 11.8 MB | Fast | Better |
| `yolov8m-seg` | 27.3 MB | Medium | Best |
| `yolov8l-seg` | 46.0 MB | Slow | Excellent |
| `yolov8x-seg` | 71.8 MB | Slowest | Top |

### YOLOv11 Segmentation

| Model | Size | Speed | Accuracy |
|:------|:----:|:-----:|:--------:|
| `yolo11n-seg` | 2.9 MB | Fastest | Good |
| `yolo11s-seg` | 10.1 MB | Fast | Better |
| `yolo11m-seg` | 22.4 MB | Medium | Best |
| `yolo11l-seg` | 27.6 MB | Slow | Excellent |
| `yolo11x-seg` | 62.1 MB | Slowest | Top |

---

## Google Colab

For GPU-accelerated training, use Google Colab.

<details>
<summary><strong>Quick Setup (Click to expand)</strong></summary>

```python
#@title SegMeow - Quick Setup

# Install
!pip install ultralytics -q

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Config - EDIT THESE
DRIVE_DATASET_PATH = "/content/drive/MyDrive/your_dataset"
COCO_JSON = "coco.json"
EPOCHS = 100
MODEL_VERSION = "yolov8"  # "yolov8" or "yolo11"
MODEL_SIZE = "n"          # n, s, m, l, x

#-------------------------------------------
import json, os, yaml
from ultralytics import YOLO

# Copy dataset
!cp -r "{DRIVE_DATASET_PATH}" /content/dataset
os.chdir("/content")

# Convert COCO to YOLO
def convert_coco():
    with open(f"dataset/{COCO_JSON}") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    labels_dir = "dataset/labels/train"
    os.makedirs(labels_dir, exist_ok=True)

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]
        seg = ann["segmentation"][0]
        yolo = [str(ann["category_id"])]
        for i in range(0, len(seg), 2):
            yolo.extend([str(seg[i]/w), str(seg[i+1]/h)])

        fname = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".txt"
        with open(f"{labels_dir}/{fname}", "a") as f:
            f.write(" ".join(yolo) + "\n")

    cats = sorted(coco.get("categories", []), key=lambda x: x["id"])
    return [c["name"] for c in cats]

classes = convert_coco()
print(f"Classes: {classes}")

# Create data.yaml
with open("data.yaml", "w") as f:
    yaml.dump({
        "path": "/content/dataset",
        "train": "images/train",
        "val": "images/train",
        "nc": len(classes),
        "names": classes
    }, f)

# Train
model = YOLO(f"{MODEL_VERSION}{MODEL_SIZE}-seg.pt")
model.train(data="data.yaml", epochs=EPOCHS, imgsz=640, batch=16, device=0)

# Save to Drive
!cp runs/segment/train/weights/best.pt "{DRIVE_DATASET_PATH}/best.pt"
print("Model saved to Drive!")
```

</details>

<details>
<summary><strong>Step-by-Step Guide (Click to expand)</strong></summary>

### 1. Enable GPU
`Runtime` → `Change runtime type` → Select `GPU`

### 2. Install Dependencies
```python
!pip install ultralytics
```

### 3. Upload Dataset
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/your_dataset" /content/dataset
```

### 4. Train Model
```python
from ultralytics import YOLO

# YOLOv8
model = YOLO("yolov8n-seg.pt")

# Or YOLOv11
# model = YOLO("yolo11n-seg.pt")

model.train(data="data.yaml", epochs=100, imgsz=640, batch=16, device=0)
```

### 5. Download Model
```python
from google.colab import files
files.download("runs/segment/train/weights/best.pt")
```

</details>

---

## Tips

| Scenario | Recommendation |
|----------|----------------|
| Small dataset (<100 images) | Use **Nano** model to avoid overfitting |
| Large dataset (>1000 images) | Use **Medium/Large** for better accuracy |
| Out of memory error | Reduce batch size to 8 or 4 |
| Want better results | Increase epochs (100-300) |
| YOLOv8 vs YOLOv11 | YOLOv11 is newer, YOLOv8 is more stable |

---

## Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/ultralytics-YOLO-00FFFF?style=for-the-badge" alt="Ultralytics">
</p>

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/sohanurislamshuvo">sohanurislamshuvo</a>
</p>
