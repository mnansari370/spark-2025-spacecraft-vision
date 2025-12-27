import os
import json
import torch
from PIL import Image
from torchvision.transforms import functional as F
from src.core import YAMLConfig

# Config and 40-epoch weights

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_spark_40ep.yml"
REPO_ROOT = THIS_DIR.parents[2]
WEIGHTS_PATH = REPO_ROOT / "checkpoints" / "detection" / "detection_model" / "best.pth"


# Test set and output json (new name!)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = YAMLConfig(str(CONFIG_PATH))
    model = cfg.model
    model.to(device)

    print("Loading weights:", WEIGHTS_PATH)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    files = sorted([f for f in os.listdir(str(TEST_DIR)) if f.endswith(".jpg")])
    print(f"Found {len(files)} test images.")

    results = []

    for i, fname in enumerate(files, 1):
        img_path = os.path.join(str(TEST_DIR), fname)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        img_resized = img.resize((640, 640))
        img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]

        probs = logits.sigmoid().max(dim=-1)
        scores = probs.values
        labels = probs.indices

        detections = []
        for s, lab, box in zip(scores, labels, boxes):
            s_val = float(s.item())
            if s_val < 0.01:
                continue

            lab = int(lab.item())
            cx, cy, w, h = box.tolist()

            detections.append({
                "category": CLASSES[lab],
                "score": s_val,
                "bbox": [float(cx), float(cy), float(w), float(h)],
            })

        results.append({"image_id": fname, "detections": detections})

        if i % 200 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] done")

    os.makedirs(os.path.dirname(str(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)

    print("Saved:", OUT_JSON)

if __name__ == "__main__":
    main()
