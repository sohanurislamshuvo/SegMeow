import streamlit as st
import os
import json
import yaml
import cv2
from utils.coco_to_yolo_seg import coco_to_yolo_seg
from utils.trainer import train_model
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧠 Label Studio → YOLO Segmentation Trainer")

def extract_classes_from_coco(coco_path):
    """Extract class names from COCO JSON categories."""
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    categories = coco.get("categories", [])
    # Sort by id to ensure correct order
    categories = sorted(categories, key=lambda x: x["id"])
    return [cat["name"] for cat in categories]

# -------------------------
# Upload Section
# -------------------------
st.header("1️⃣ Upload Label Studio COCO Segmentation")

coco_file = st.file_uploader("Upload COCO JSON", type=["json"])
image_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)

if coco_file and image_files:
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)

    for img in image_files:
        with open(f"dataset/images/train/{img.name}", "wb") as f:
            f.write(img.getbuffer())

    with open("coco.json", "wb") as f:
        f.write(coco_file.getbuffer())

    coco_to_yolo_seg(
        "coco.json",
        "dataset/images/train",
        "dataset/labels/train"
    )

    st.success("✅ COCO converted to YOLO Segmentation")

# -------------------------
# Data.yaml
# -------------------------
st.header("2️⃣ Dataset Config")

# Auto-load classes from COCO JSON if exists
default_classes = ""
if os.path.isfile("coco.json"):
    coco_classes = extract_classes_from_coco("coco.json")
    if coco_classes:
        default_classes = ", ".join(coco_classes)
        st.info(f"**Classes from COCO:** {default_classes}")

class_names = st.text_input("Class names (comma separated)", default_classes)

if st.button("Generate data.yaml"):
    names = [c.strip() for c in class_names.split(",") if c.strip()]

    if not names:
        st.error("Please enter at least one class name.")
    else:
        data = {
            "path": "dataset",
            "train": "images/train",
            "val": "images/train",
            "nc": len(names),
            "names": names
        }

        with open("dataset/data.yaml", "w") as f:
            yaml.dump(data, f)

        st.success("✅ data.yaml created")

# -------------------------
# Training
# -------------------------
st.header("3️⃣ Train / Resume YOLO Segmentation")

# Model version and size selection
model_version = st.selectbox(
    "Model Version",
    ["YOLOv8", "YOLOv11"],
    help="Choose between YOLOv8 (stable) or YOLOv11 (latest)"
)

# Define available model sizes for each version
model_sizes = {
    "YOLOv8": {
        "Nano (fastest)": "yolov8n-seg.pt",
        "Small": "yolov8s-seg.pt",
        "Medium": "yolov8m-seg.pt",
        "Large": "yolov8l-seg.pt",
        "XLarge (most accurate)": "yolov8x-seg.pt"
    },
    "YOLOv11": {
        "Nano (fastest)": "yolo11n-seg.pt",
        "Small": "yolo11s-seg.pt",
        "Medium": "yolo11m-seg.pt",
        "Large": "yolo11l-seg.pt",
        "XLarge (most accurate)": "yolo11x-seg.pt"
    }
}

model_size = st.selectbox(
    "Model Size",
    list(model_sizes[model_version].keys()),
    help="Larger models are more accurate but slower to train"
)

weights = model_sizes[model_version][model_size]
st.info(f"Selected model: `{weights}`")

epochs = st.slider("Epochs", 1, 200, 50)
imgsz = st.selectbox("Image Size", [640, 1024])

if st.button("Start Training"):
    train_model("dataset/data.yaml", weights, epochs, imgsz)
    st.success("🎉 Training started")

# -------------------------
# Inference
# -------------------------
st.header("4️⃣ Polygon Detection")

def get_latest_model():
    """Find the latest trained model."""
    runs_dir = "runs/segment"
    if not os.path.exists(runs_dir):
        return None
    # Get all train directories sorted by modification time
    train_dirs = [d for d in os.listdir(runs_dir) if d.startswith("train")]
    if not train_dirs:
        return None
    # Sort by modification time (newest first)
    train_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)), reverse=True)
    best_model = os.path.join(runs_dir, train_dirs[0], "weights", "best.pt")
    if os.path.exists(best_model):
        return best_model
    return None

latest_model = get_latest_model()
if latest_model:
    st.info(f"Using model: `{latest_model}`")

infer_img = st.file_uploader("Upload image for inference", type=["jpg", "png"])

if infer_img:
    if not latest_model:
        st.error("No trained model found. Please train a model first.")
    else:
        with open("infer.jpg", "wb") as f:
            f.write(infer_img.getbuffer())

        model = YOLO(latest_model)
        results = model("infer.jpg")

        # Plot with better visibility settings
        result_img = results[0].plot(
            line_width=3,        # Thicker lines
            font_size=1.0,       # Larger font
            conf=True,           # Show confidence
            labels=True          # Show class labels
        )

        # Add detection count overlay on image
        count = len(results[0].masks) if results[0].masks is not None else 0
        cv2.putText(
            result_img,
            f"Detections: {count}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(infer_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(result_img, caption="Detection Result", use_container_width=True)

        # Show detection details
        if results[0].masks is not None:
            st.success(f"Detected {len(results[0].masks)} polygon(s)")
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = results[0].names[cls]
                st.write(f"- **{cls_name}**: {conf:.2%} confidence")
        else:
            st.warning("No polygons detected in this image.")
