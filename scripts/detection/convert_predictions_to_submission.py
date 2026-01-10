import json
import csv
import math
from pathlib import Path
from PIL import Image

# repo root: scripts/detection/ -> scripts/ -> repo
REPO_ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = REPO_ROOT / "inference_results" / "detection" / "detection_test_predictions.json"
TEST_DIR  = REPO_ROOT / "data" / "spark-2024-detection-test" / "images"
OUT_CSV   = REPO_ROOT / "inference_results" / "detection" / "detection.csv"

SCORE_FLOOR = 0.30
SHRINK = 0.97
SIGMA_CENTER = 0.35
AREA_ALPHA = 0.30

DUMMY_CLASS = "Cheops"


def pick_best_detection(detections):
    if not detections:
        return None

    filtered = None
    for thr in [SCORE_FLOOR, 0.10, 0.01]:
        cand = [d for d in detections if float(d.get("score", 0.0)) >= thr]
        if cand:
            filtered = cand
            break

    if filtered is None:
        return None

    best = None
    best_val = -1e18

    for d in filtered:
        s = float(d["score"])
        cx, cy, w, h = d["bbox"]

        cx = float(cx)
        cy = float(cy)
        w = max(1e-6, min(float(w), 1.0))
        h = max(1e-6, min(float(h), 1.0))

        area = w * h
        dx = cx - 0.5
        dy = cy - 0.5
        dist2 = dx * dx + dy * dy

        center_prior = math.exp(-dist2 / (2.0 * SIGMA_CENTER * SIGMA_CENTER))
        score = s * (area ** AREA_ALPHA) * center_prior

        if score > best_val:
            best_val = score
            best = d

    return best


def shrink_box(cx, cy, w, h):
    if SHRINK >= 0.9999:
        return cx, cy, w, h
    w = max(w * SHRINK, 1e-4)
    h = max(h * SHRINK, 1e-4)
    return cx, cy, w, h


def cxcywh_to_xyxy(cx, cy, w, h, W, H):
    xmin = int((cx - w / 2) * W)
    ymin = int((cy - h / 2) * H)
    xmax = int((cx + w / 2) * W)
    ymax = int((cy + h / 2) * H)

    xmin = max(0, min(xmin, W - 1))
    ymin = max(0, min(ymin, H - 1))
    xmax = max(0, min(xmax, W - 1))
    ymax = max(0, min(ymax, H - 1))

    if xmax <= xmin:
        xmax = min(W - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(H - 1, ymin + 1)

    return xmin, ymin, xmax, ymax


def centered_dummy_box(W, H):
    bw = max(2, int(W * 0.05))
    bh = max(2, int(H * 0.05))
    cx = W * 0.5
    cy = H * 0.5

    xmin = int(cx - bw / 2)
    ymin = int(cy - bh / 2)
    xmax = int(cx + bw / 2)
    ymax = int(cy + bh / 2)

    xmin = max(0, min(xmin, W - 1))
    ymin = max(0, min(ymin, H - 1))
    xmax = max(0, min(xmax, W - 1))
    ymax = max(0, min(ymax, H - 1))

    if xmax <= xmin:
        xmax = min(W - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(H - 1, ymin + 1)

    return xmin, ymin, xmax, ymax


def main():
    data = json.loads(JSON_PATH.read_text())

    pred_map = {}
    for item in data:
        fname = item.get("image_id")
        if fname is not None:
            pred_map[fname] = item.get("detections", [])

    # supports flat or nested folders
    test_files = sorted([p.relative_to(TEST_DIR).as_posix() for p in TEST_DIR.rglob("*.jpg")])
    print(f"[INFO] Found {len(test_files)} test images.")

    rows = []
    dummy_count = 0

    for fname in test_files:
        img_path = TEST_DIR / fname
        with Image.open(img_path) as im:
            W, H = im.size

        detections = pred_map.get(fname, [])
        best = pick_best_detection(detections)

        if best is None:
            dummy_count += 1
            cls = DUMMY_CLASS
            xmin, ymin, xmax, ymax = centered_dummy_box(W, H)
        else:
            cls = best["category"]
            cx, cy, w, h = best["bbox"]
            cx, cy, w, h = shrink_box(float(cx), float(cy), float(w), float(h))
            xmin, ymin, xmax, ymax = cxcywh_to_xyxy(cx, cy, w, h, W, H)

        bbox_str = f"({xmin}, {ymin}, {xmax}, {ymax})"
        rows.append([fname, cls, bbox_str])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "bbox"])
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} predictions + 1 header = {len(rows) + 1}")
    print(f"[INFO] Dummy boxes used: {dummy_count}")

    if len(test_files) == 20000 and (len(rows) + 1) != 20001:
        raise RuntimeError("Row count is NOT 20001!")
    if len(rows) != len(test_files):
        raise RuntimeError("Mismatch between images and rows!")


if __name__ == "__main__":
    main()
