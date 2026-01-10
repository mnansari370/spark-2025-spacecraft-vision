import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# Helpers
# -------------------------
def find_repo_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "models").exists() and (d / "scripts").exists():
            return d
    raise RuntimeError(f"Could not find repo root from {start}")

def cxcywh_to_xyxy(cx, cy, w, h, W, H):
    # Input is normalized cx,cy,w,h in [0,1] (typical for DETR)
    # Convert to pixel xyxy
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return [x1, y1, x2, y2]

def clamp_box(b, W, H):
    x1, y1, x2, y2 = b
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    # ensure ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [x1, y1, x2, y2]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_predictions(json_path: Path):
    data = json.loads(json_path.read_text())
    # expected: list of {"image_id":..., "detections":[{"category","score","bbox":[cx,cy,w,h]}...]}
    if not isinstance(data, list):
        raise ValueError("Detection JSON must be a list of per-image predictions.")
    return data

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_thr", type=float, default=0.30, help="Score threshold for analysis/visualization")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vis_n", type=int, default=20, help="Number of images to visualize")
    ap.add_argument("--topk", type=int, default=30, help="Top-K images table sizes")
    ap.add_argument("--max_vis_per_image", type=int, default=25, help="Max boxes to draw per image")
    ap.add_argument("--thr_sweep", type=str, default="0.05,0.1,0.2,0.3,0.4,0.5",
                    help="Comma thresholds for sweep plots")
    args = ap.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    fig_dir = repo / "reports" / "figures"
    tab_dir = repo / "reports" / "tables"
    safe_mkdir(fig_dir); safe_mkdir(tab_dir)

    json_path = repo / "inference_results" / "detection" / "detection_test_predictions.json"
    img_dir = repo / "data" / "spark-2024-detection-test" / "images"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing detection JSON: {json_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing detection test images dir: {img_dir}")

    preds = load_predictions(json_path)
    print("Loaded predictions:", len(preds))

    thr = args.score_thr
    sweep = [float(x.strip()) for x in args.thr_sweep.split(",") if x.strip()]

    # ---- Aggregate stats ----
    dets_per_image = []
    score_list = []
    cls_count = Counter()
    cls_score_sum = defaultdict(float)
    cls_score_n = defaultdict(int)

    # for top-k tables
    img_total_dets = []   # (n_dets, image_id)
    img_mean_score = []   # (mean_score, image_id)

    for row in preds:
        image_id = row.get("image_id")
        dets = row.get("detections", [])
        dets_f = [d for d in dets if float(d.get("score", 0.0)) >= thr]

        dets_per_image.append(len(dets_f))
        if dets_f:
            scores = [float(d["score"]) for d in dets_f]
            img_mean_score.append((float(np.mean(scores)), image_id))
        else:
            img_mean_score.append((0.0, image_id))

        img_total_dets.append((len(dets_f), image_id))

        for d in dets_f:
            c = d.get("category", "UNK")
            s = float(d.get("score", 0.0))
            cls_count[c] += 1
            score_list.append(s)
            cls_score_sum[c] += s
            cls_score_n[c] += 1

    # ---- Plot: per-class avg score ----
    classes = sorted(cls_count.keys())
    avg_scores = [(cls_score_sum[c] / max(1, cls_score_n[c])) for c in classes]
    counts = [cls_count[c] for c in classes]

    plt.figure()
    plt.bar(range(len(classes)), avg_scores)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.ylabel("Average confidence")
    plt.title(f"Detection: Avg confidence per class (thr={thr:.2f})")
    outp = fig_dir / "detection_avg_conf_per_class.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    print("Wrote:", outp)

    # ---- Plot: threshold sweep (total detections) ----
    sweep_counts = []
    for t in sweep:
        total = 0
        for row in preds:
            dets = row.get("detections", [])
            total += sum(1 for d in dets if float(d.get("score", 0.0)) >= t)
        sweep_counts.append(total)

    plt.figure()
    plt.plot(sweep, sweep_counts, marker="o")
    plt.xlabel("Score threshold")
    plt.ylabel("Total detections")
    plt.title("Detection: Total detections vs score threshold")
    outp = fig_dir / "detection_thr_sweep_total_dets.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    print("Wrote:", outp)

    # ---- Table: top images by #dets ----
    img_total_dets.sort(reverse=True, key=lambda x: x[0])
    top_dets = img_total_dets[: args.topk]
    out_csv = tab_dir / "detection_top_images_by_count.csv"
    with out_csv.open("w") as f:
        f.write("rank,image_id,num_detections\n")
        for i, (n, imgid) in enumerate(top_dets, 1):
            f.write(f"{i},{imgid},{n}\n")
    print("Wrote:", out_csv)

    # ---- Table: top images by mean score ----
    img_mean_score.sort(reverse=True, key=lambda x: x[0])
    top_score = img_mean_score[: args.topk]
    out_csv = tab_dir / "detection_top_images_by_mean_score.csv"
    with out_csv.open("w") as f:
        f.write("rank,image_id,mean_score\n")
        for i, (m, imgid) in enumerate(top_score, 1):
            f.write(f"{i},{imgid},{m:.6f}\n")
    print("Wrote:", out_csv)

    # ---- Visualization grid: sample images with boxes ----
    random.seed(args.seed)
    candidates = [r["image_id"] for r in preds]
    random.shuffle(candidates)
    sample_ids = candidates[: args.vis_n]

    # try to use a default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    vis_imgs = []
    for imgid in sample_ids:
        img_path = img_dir / imgid
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)

        row = next((r for r in preds if r.get("image_id") == imgid), None)
        dets = (row.get("detections", []) if row else [])
        dets = [d for d in dets if float(d.get("score", 0.0)) >= thr]
        dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)[: args.max_vis_per_image]

        for d in dets:
            c = d.get("category", "UNK")
            s = float(d.get("score", 0.0))
            bbox = d.get("bbox", [0, 0, 0, 0])
            cx, cy, w, h = [float(x) for x in bbox]
            b = cxcywh_to_xyxy(cx, cy, w, h, W, H)
            b = clamp_box(b, W, H)
            draw.rectangle(b, width=2)
            txt = f"{c}:{s:.2f}"
            x1, y1, x2, y2 = b
            draw.text((x1 + 2, max(0, y1 - 12)), txt, font=font)

        vis_imgs.append(im)

    if not vis_imgs:
        print("No images were visualized (check paths).")
        return

    # Make a grid
    n = len(vis_imgs)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    thumb_w = 480
    thumb_h = int(thumb_w * 0.75)

    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))
    for i, im in enumerate(vis_imgs):
        im = im.resize((thumb_w, thumb_h))
        r = i // cols
        c = i % cols
        grid.paste(im, (c * thumb_w, r * thumb_h))

    outp = fig_dir / "detection_vis_grid.png"
    grid.save(outp)
    print("Wrote:", outp)

    # ---- Summary txt ----
    out_txt = tab_dir / "detection_extras_summary.txt"
    with out_txt.open("w") as f:
        f.write(f"Detection extras summary\n")
        f.write(f"Score threshold used: {thr}\n")
        f.write(f"Images in JSON: {len(preds)}\n")
        f.write(f"Total detections (thr): {sum(dets_per_image)}\n")
        f.write(f"Mean dets/image (thr): {float(np.mean(dets_per_image)):.4f}\n")
        if score_list:
            f.write(f"Mean score (thr): {float(np.mean(score_list)):.4f}\n")
            f.write(f"Median score (thr): {float(np.median(score_list)):.4f}\n")
        f.write("Threshold sweep:\n")
        for t, c in zip(sweep, sweep_counts):
            f.write(f"  thr={t:.2f} -> total_dets={c}\n")
    print("Wrote:", out_txt)

if __name__ == "__main__":
    main()
