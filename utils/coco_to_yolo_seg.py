import json
import os

def coco_to_yolo_seg(coco_json, images_dir, labels_dir):
    coco = json.load(open(coco_json))

    images = {img["id"]: img for img in coco["images"]}

    # Clear old labels to prevent duplicates
    if os.path.exists(labels_dir):
        for f in os.listdir(labels_dir):
            file_path = os.path.join(labels_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.makedirs(labels_dir, exist_ok=True)

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]
        # Label Studio uses 0-indexed category IDs, YOLO also uses 0-indexed
        cls = ann["category_id"]

        seg = ann["segmentation"][0]

        yolo = [str(cls)]
        for i in range(0, len(seg), 2):
            yolo.append(str(seg[i] / w))
            yolo.append(str(seg[i+1] / h))

        # Extract just the filename (Label Studio exports full paths)
        file_name = os.path.basename(img["file_name"])
        label_name = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "a") as f:
            f.write(" ".join(yolo) + "\n")
