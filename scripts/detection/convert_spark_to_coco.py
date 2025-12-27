import os
import json
import ast

import pandas as pd
from PIL import Image

"""
Convert SPARK-2024 detection CSV annotations (train/val)
to COCO-style JSON files.
"""

CLASS_NAMES = [
    "Cheops",
    "LisaPathfinder",
    "ObservationSat1",
    "Proba2",
    "Proba3",
    "Proba3ocs",
    "Smart1",
    "Soho",
    "VenusExpress",
    "XMM Newton",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def convert_split(root, csv_name, split, out_json):
    """
    Convert one split (train or val) from CSV to COCO JSON.

    root:      path to spark-2024-train-val
    csv_name:  'train.csv' or 'val.csv'
    split:     'train' or 'val'
    out_json:  output JSON path
    """
    csv_path = os.path.join(root, csv_name)
    df = pd.read_csv(csv_path)

    images = []
    annotations = []
    categories = [{"id": cid, "name": name} for name, cid in CLASS_TO_ID.items()]

    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1

    for idx, row in df.iterrows():
        img_name = row["Image name"]
        class_name = row["Class"]
        bbox_str = row["Bounding box"]

        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{class_name}' in row {idx}")

        # "(xmin, ymin, xmax, ymax)" → four floats
        xmin, ymin, xmax, ymax = ast.literal_eval(bbox_str)
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        w = xmax - xmin
        h = ymax - ymin

        rel_path = os.path.join(class_name, split, img_name)
        img_key = rel_path

        if img_key not in image_id_map:
            image_id = next_image_id
            image_id_map[img_key] = image_id
            next_image_id += 1

            full_img_path = os.path.join(root, "images", rel_path)
            if not os.path.exists(full_img_path):
                raise FileNotFoundError(f"Image not found: {full_img_path}")

            with Image.open(full_img_path) as im:
                width, height = im.size

            images.append(
                {
                    "id": image_id,
                    "file_name": rel_path.replace("\\", "/"),
                    "width": width,
                    "height": height,
                }
            )
        else:
            image_id = image_id_map[img_key]

        annotations.append(
            {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": CLASS_TO_ID[class_name],
                "bbox": [xmin, ymin, w, h],  # COCO format
                "area": float(w * h),
                "iscrowd": 0,
            }
        )
        next_ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"Saved {out_json}: {len(images)} images, {len(annotations)} annotations")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(project_root, "data", "spark-2024-train-val")

    out_dir = os.path.join(project_root, "data", "annotations")
    os.makedirs(out_dir, exist_ok=True)

    convert_split(
        root=root,
        csv_name="train.csv",
        split="train",
        out_json=os.path.join(out_dir, "spark_train.json"),
    )

    convert_split(
        root=root,
        csv_name="val.csv",
        split="val",
        out_json=os.path.join(out_dir, "spark_val.json"),
    )


if __name__ == "__main__":
    main()

