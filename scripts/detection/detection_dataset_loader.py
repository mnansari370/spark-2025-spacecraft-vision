import os
import ast
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

"""
PyTorch dataset for the SPARK-2024 detection task.

It reads:
- a CSV file (train.csv or val.csv)
- images from: root/images/<Class>/<split>/<Image name>

Columns used from the CSV:
  "Image name", "Class", "Bounding box"
where "Bounding box" is "(xmin, ymin, xmax, ymax)".
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


class SparkDetectionDataset(Dataset):
    def __init__(self, root, csv_file, split="train", transforms=None):
        """
        root:       path to 'spark-2024-train-val'
        csv_file:   path to train.csv or val.csv
        split:      'train' or 'val'
        transforms: optional callable (img, target) -> (img, target)
        """
        self.root = root
        self.csv_file = csv_file
        self.split = split
        self.transforms = transforms

        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["Image name"]
        class_name = row["Class"]

        bbox_str = row["Bounding box"]
        xmin, ymin, xmax, ymax = ast.literal_eval(bbox_str)
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

        img_path = os.path.join(
            self.root,
            "images",
            class_name,
            self.split,
            img_name,
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        if class_name not in CLASS_TO_ID:
            raise KeyError(
                f"Class name '{class_name}' not in CLASS_TO_ID. "
                f"Available: {list(CLASS_TO_ID.keys())}"
            )

        labels = torch.tensor([CLASS_TO_ID[class_name]], dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

