import os
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as F


THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    # Walk upward until we find a folder that looks like your repo root
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "data").exists() and (d / "models").exists():
            return d
    # fallback: assume spark_project is one of the parents
    for d in [start] + list(start.parents):
        if d.name == "spark_project":
            return d
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)

# Ensure RT-DETRv2 code is importable even if PYTHONPATH isn't set
# (you may need to adjust this if your repo structure is different)
sys.path.insert(0, str(THIS_DIR))

from src.core import YAMLConfig  # noqa: E402


CONFIG_PATH = THIS_DIR / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_spark_40ep.yml"
WEIGHTS_PATH = REPO_ROOT / "checkpoints" / "detection" / "detection_model" / "best.pth"
TEST_DIR = REPO_ROOT / "data" / "spark-2024-detection-test" / "images"
OUT_JSON = REPO_ROOT / "inference_results" / "detection" / "detection_test_predictions.json"

CLASSES = [
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


def main():
    # --- sanity checks
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("REPO_ROOT:", REPO_ROOT)
    print("CONFIG:", CONFIG_PATH)
    print("WEIGHTS:", WEIGHTS_PATH)
    print("TEST_DIR:", TEST_DIR)
    print("OUT_JSON:", OUT_JSON)

    cfg = YAMLConfig(str(CONFIG_PATH))
    model = cfg.model.to(device)

    print("Loading weights...")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])
    print(f"Found {len(files)} test images.")

    results = []

    for i, fname in enumerate(files, 1):
        img_path = TEST_DIR / fname
        img = Image.open(img_path).convert("RGB")

        # Model expects 640x640 in your current pipeline
        img_resized = img.resize((640, 640))
        img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        logits = outputs["pred_logits"][0]  # [num_queries, num_classes]
        boxes = outputs["pred_boxes"][0]    # [num_queries, 4] (likely cx,cy,w,h normalized)

        probs = logits.sigmoid().max(dim=-1)
        scores = probs.values
        labels = probs.indices

        detections = []
        for s, lab, box in zip(scores, labels, boxes):
            s_val = float(s.item())
            if s_val < 0.01:
                continue

            lab = int(lab.item())
            cx, cy, w, h = [float(x) for x in box.tolist()]

            # Keep as model-format bbox (normalized cx,cy,w,h)
            # Convert to float for JSON serialization
            detections.append(
                {
                    "category": CLASSES[lab],
                    "score": s_val,
                    "bbox": [cx, cy, w, h],
                }
            )

        results.append({"image_id": fname, "detections": detections})

        if i % 200 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] done")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)

    print("Saved:", OUT_JSON)


if __name__ == "__main__":
    main()
