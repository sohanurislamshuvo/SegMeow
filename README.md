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

Train your segmentation model with free GPU on Google Colab.

### Prerequisites

Before starting, make sure you have:
- A Google account
- Your dataset uploaded to Google Drive with this structure:
  ```
  your_dataset/
  ├── coco.json          # COCO format annotations from Label Studio
  └── images/
      └── train/         # Your training images
  ```

---

### Step 1: Enable GPU Runtime

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Go to `Runtime` → `Change runtime type`
4. Select `GPU` (T4 recommended) → Click `Save`

---

### Step 2: Install Dependencies

```python
# Install required packages
!pip install ultralytics pyyaml -q
```

---

### Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Step 4: Copy Dataset to Colab

```python
import shutil
import os

# CHANGE THIS to your dataset path in Google Drive
DRIVE_DATASET_PATH = "/content/drive/MyDrive/your_dataset"

# Copy dataset to Colab runtime (faster training)
shutil.copytree(DRIVE_DATASET_PATH, "/content/dataset")

print("Dataset copied successfully!")
print("Files:", os.listdir("/content/dataset"))
```

---

### Step 5: Convert COCO to YOLO Format

```python
import json
import os

def coco_to_yolo_seg(coco_json_path, labels_output_dir):
    """Convert COCO JSON annotations to YOLO segmentation format"""

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Create lookup for images
    images = {img["id"]: img for img in coco["images"]}

    # Create labels directory
    os.makedirs(labels_output_dir, exist_ok=True)

    # Clear existing labels
    for f in os.listdir(labels_output_dir):
        os.remove(os.path.join(labels_output_dir, f))

    # Convert each annotation
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]
        seg = ann["segmentation"][0]

        # Normalize coordinates
        yolo_line = [str(ann["category_id"])]
        for i in range(0, len(seg), 2):
            yolo_line.append(str(seg[i] / w))      # x normalized
            yolo_line.append(str(seg[i + 1] / h))  # y normalized

        # Write to label file
        img_name = os.path.basename(img["file_name"])
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_output_dir, label_name)

        with open(label_path, "a") as f:
            f.write(" ".join(yolo_line) + "\n")

    print(f"Converted {len(coco['annotations'])} annotations to YOLO format")
    return coco

# Run conversion
coco_data = coco_to_yolo_seg(
    coco_json_path="/content/dataset/coco.json",
    labels_output_dir="/content/dataset/labels/train"
)
```

---

### Step 6: Create Dataset Configuration (data.yaml)

```python
import yaml

# Extract class names from COCO
categories = sorted(coco_data.get("categories", []), key=lambda x: x["id"])
class_names = [cat["name"] for cat in categories]

print(f"Found {len(class_names)} classes: {class_names}")

# Create data.yaml
data_config = {
    "path": "/content/dataset",
    "train": "images/train",
    "val": "images/train",  # Using train as val (change if you have separate val set)
    "nc": len(class_names),
    "names": class_names
}

with open("/content/data.yaml", "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("\ndata.yaml created!")
print(yaml.dump(data_config, default_flow_style=False))
```

---

### Step 7: Train the Model

```python
from ultralytics import YOLO

# ============ CONFIGURATION ============
MODEL = "yolov8n-seg.pt"  # Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
                          #          yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
EPOCHS = 100              # Number of training epochs
IMG_SIZE = 640            # Image size (640 or 1024)
BATCH_SIZE = 16           # Reduce to 8 if out of memory
# =======================================

# Load model
model = YOLO(MODEL)

# Start training
results = model.train(
    data="/content/data.yaml",
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=0,         # GPU
    workers=2,
    project="runs",
    name="train",
    exist_ok=True
)

print("\nTraining complete!")
print(f"Best model saved at: runs/train/weights/best.pt")
```

---

### Step 8: Test Inference

```python
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import os

# Load trained model
model = YOLO("runs/train/weights/best.pt")

# Get a test image (first image from training set)
test_images_dir = "/content/dataset/images/train"
test_image = os.path.join(test_images_dir, os.listdir(test_images_dir)[0])

# Run inference
results = model(test_image)

# Display result with segmentation masks
result_img = results[0].plot(line_width=2)
cv2_imshow(result_img)

# Print detection details
for r in results:
    if r.masks is not None:
        print(f"Detected {len(r.masks)} objects")
```

---

### Step 9: Save Model to Google Drive

```python
import shutil

# CHANGE THIS to your desired save location
SAVE_PATH = "/content/drive/MyDrive/trained_model.pt"

# Copy best weights to Drive
shutil.copy("runs/train/weights/best.pt", SAVE_PATH)

print(f"Model saved to: {SAVE_PATH}")
```

---

### Step 10: Download Model (Optional)

```python
from google.colab import files

# Download to your computer
files.download("runs/train/weights/best.pt")
```

---

### Quick Copy-Paste Version

<details>
<summary><strong>All-in-One Script (Click to expand)</strong></summary>

```python
#@title SegMeow Training - All in One Cell

# ============ CONFIGURATION - EDIT THESE ============
DRIVE_DATASET_PATH = "/content/drive/MyDrive/your_dataset"  # Your dataset in Drive
MODEL = "yolov8n-seg.pt"    # Model to use
EPOCHS = 100                 # Training epochs
IMG_SIZE = 640               # Image size
BATCH_SIZE = 16              # Batch size (reduce if OOM)
# ====================================================

# Install
!pip install ultralytics pyyaml -q

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset
import shutil, os, json, yaml
shutil.copytree(DRIVE_DATASET_PATH, "/content/dataset")

# Convert COCO to YOLO
with open("/content/dataset/coco.json") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
os.makedirs("/content/dataset/labels/train", exist_ok=True)

for ann in coco["annotations"]:
    img = images[ann["image_id"]]
    w, h = img["width"], img["height"]
    seg = ann["segmentation"][0]
    yolo = [str(ann["category_id"])] + [str(seg[i]/w if i%2==0 else seg[i]/h) for i in range(len(seg))]
    label_name = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".txt"
    with open(f"/content/dataset/labels/train/{label_name}", "a") as f:
        f.write(" ".join(yolo) + "\n")

# Create data.yaml
categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
class_names = [c["name"] for c in categories]
with open("/content/data.yaml", "w") as f:
    yaml.dump({"path": "/content/dataset", "train": "images/train", "val": "images/train",
               "nc": len(class_names), "names": class_names}, f)

print(f"Classes: {class_names}")

# Train
from ultralytics import YOLO
model = YOLO(MODEL)
model.train(data="/content/data.yaml", epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=0)

# Save to Drive
shutil.copy("runs/segment/train/weights/best.pt", f"{DRIVE_DATASET_PATH}/best.pt")
print(f"\nModel saved to: {DRIVE_DATASET_PATH}/best.pt")
```

</details>

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 8 or 4 |
| `No GPU available` | Go to Runtime → Change runtime type → Select GPU |
| `File not found` | Check your `DRIVE_DATASET_PATH` is correct |
| `No annotations converted` | Ensure `coco.json` has `segmentation` field (polygon annotations) |
| `Training too slow` | Use a smaller model (nano or small) |

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
