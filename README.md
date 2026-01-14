# YOLO Polygon Segmentation Trainer

A Streamlit application for training YOLOv8 and YOLOv11 segmentation models using Label Studio COCO annotations.

## Features

- Convert Label Studio COCO JSON to YOLO segmentation format
- Train **YOLOv8** or **YOLOv11** segmentation models
- Choose from 5 model sizes (Nano → XLarge)
- Run inference and visualize polygon detections
- Auto-extract class names from COCO annotations

## Local Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

## Project Structure

```
polygon/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── coco.json                 # COCO format annotations (from Label Studio)
├── dataset/
│   ├── images/train/         # Training images
│   └── labels/train/         # YOLO format labels
├── utils/
│   ├── coco_to_yolo_seg.py   # COCO to YOLO converter
│   └── trainer.py            # Training wrapper
└── runs/segment/             # Training outputs
```

---

## Train on Google Colab (GPU)

For faster training, use Google Colab with GPU acceleration.

### Step 1: Create a New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Enable GPU

1. Click `Runtime` → `Change runtime type`
2. Select `GPU` (T4 recommended)
3. Click `Save`

### Step 3: Install Dependencies

```python
!pip install ultralytics
```

### Step 4: Upload Your Dataset

Option A - Upload from local:
```python
from google.colab import files

# Upload coco.json
uploaded = files.upload()

# Upload images (or use zip)
# For multiple images, zip them first and upload the zip
```

Option B - Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy your dataset from Drive
!cp -r "/content/drive/MyDrive/your_dataset_folder" /content/dataset
```

### Step 5: Convert COCO to YOLO Format

```python
import json
import os

def coco_to_yolo_seg(coco_json, images_dir, labels_dir):
    with open(coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    os.makedirs(labels_dir, exist_ok=True)

    # Clear old labels
    for f_name in os.listdir(labels_dir) if os.path.exists(labels_dir) else []:
        file_path = os.path.join(labels_dir, f_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]
        cls = ann["category_id"]

        seg = ann["segmentation"][0]

        yolo = [str(cls)]
        for i in range(0, len(seg), 2):
            yolo.append(str(seg[i] / w))
            yolo.append(str(seg[i+1] / h))

        file_name = os.path.basename(img["file_name"])
        label_name = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "a") as f:
            f.write(" ".join(yolo) + "\n")

    print(f"Converted {len(coco['annotations'])} annotations")

# Run conversion
coco_to_yolo_seg("coco.json", "dataset/images/train", "dataset/labels/train")
```

### Step 6: Extract Class Names from COCO

```python
import json

with open("coco.json") as f:
    coco = json.load(f)

categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
class_names = [cat["name"] for cat in categories]
print("Classes:", class_names)
```

### Step 7: Create data.yaml

```python
import yaml

data = {
    "path": "/content/dataset",
    "train": "images/train",
    "val": "images/train",  # Use same for small datasets
    "nc": len(class_names),
    "names": class_names
}

with open("data.yaml", "w") as f:
    yaml.dump(data, f)

print("data.yaml created!")
```

### Step 8: Train the Model

```python
from ultralytics import YOLO

# Load pretrained model - Choose YOLOv8 or YOLOv11

# YOLOv8 options:
model = YOLO("yolov8n-seg.pt")  # nano (fastest)
# model = YOLO("yolov8s-seg.pt")  # small
# model = YOLO("yolov8m-seg.pt")  # medium
# model = YOLO("yolov8l-seg.pt")  # large
# model = YOLO("yolov8x-seg.pt")  # xlarge (most accurate)

# YOLOv11 options (latest):
# model = YOLO("yolo11n-seg.pt")  # nano (fastest)
# model = YOLO("yolo11s-seg.pt")  # small
# model = YOLO("yolo11m-seg.pt")  # medium
# model = YOLO("yolo11l-seg.pt")  # large
# model = YOLO("yolo11x-seg.pt")  # xlarge (most accurate)

# Train
results = model.train(
    data="data.yaml",
    epochs=100,           # Adjust as needed
    imgsz=640,            # Image size
    batch=16,             # Batch size (reduce if OOM)
    device=0,             # GPU
    workers=2,
    project="runs",
    name="polygon_train",
    exist_ok=True
)
```

### Step 9: Download Trained Model

```python
from google.colab import files

# Download best weights
files.download("runs/polygon_train/weights/best.pt")
```

### Step 10: Test Inference

```python
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

# Load trained model
model = YOLO("runs/polygon_train/weights/best.pt")

# Run inference
results = model("test_image.jpg")

# Display result
result_img = results[0].plot(line_width=3)
cv2_imshow(result_img)
```

---

## Complete Colab Notebook

Copy this entire code block into a Colab cell:

```python
#@title Setup and Train YOLO Segmentation Model

# Install
!pip install ultralytics -q

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Config - EDIT THESE
DRIVE_DATASET_PATH = "/content/drive/MyDrive/polygon_dataset"  # Your dataset folder
COCO_JSON = "coco.json"
EPOCHS = 100
MODEL_VERSION = "yolov8"  # "yolov8" or "yolo11"
MODEL_SIZE = "n"  # n=nano, s=small, m=medium, l=large, x=xlarge

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

    # Get classes
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

---

## Tips

- **Small dataset?** Use nano models to avoid overfitting
- **Large dataset?** Use medium/large models for better accuracy
- **Out of memory?** Reduce `batch` size (8 or 4)
- **Better results?** Increase `epochs` (100-300)
- **YOLOv8 vs YOLOv11?** YOLOv11 is newer with improved architecture, YOLOv8 is more stable

## Model Sizes

### YOLOv8 Segmentation

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n-seg | 3.4 MB | Fastest | Good |
| yolov8s-seg | 11.8 MB | Fast | Better |
| yolov8m-seg | 27.3 MB | Medium | Best |
| yolov8l-seg | 46.0 MB | Slow | Excellent |
| yolov8x-seg | 71.8 MB | Slowest | Top |

### YOLOv11 Segmentation

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo11n-seg | 2.9 MB | Fastest | Good |
| yolo11s-seg | 10.1 MB | Fast | Better |
| yolo11m-seg | 22.4 MB | Medium | Best |
| yolo11l-seg | 27.6 MB | Slow | Excellent |
| yolo11x-seg | 62.1 MB | Slowest | Top |

## License

MIT
#   S e g M e o w  
 #   S e g M e o w  
 