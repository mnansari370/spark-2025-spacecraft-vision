from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def cxcywh_norm_to_xyxy(box, W, H):
    cx, cy, w, h = box
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return [x1, y1, x2, y2]


def coco_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match_counts(gt_boxes, pred_boxes, iou_thr):
    """
    Count TP/FP/FN using greedy score-ordered matching.
    gt_boxes: list[xyxy]
    pred_boxes: list[xyxy] (already filtered by threshold)
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pi in range(len(pred_boxes)):
        best_iou = 0.0
        best_gi = -1
        for gi in range(len(gt_boxes)):
            if gi in matched_gt:
                continue
            iou = iou_xyxy(pred_boxes[pi], gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gi)
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_gt", default="data/annotations/spark_val.json")
    ap.add_argument("--pred_json", default="inference_results/detection/val_predictions.json")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--thr_min", type=float, default=0.0)
    ap.add_argument("--thr_max", type=float, default=1.0)
    ap.add_argument("--thr_step", type=float, default=0.02)
    ap.add_argument("--out_png", default="reports/figures/detection_val_threshold_sweep.png")
    ap.add_argument("--out_csv", default="reports/tables/detection_val_threshold_sweep.csv")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco_gt).read_text())
    preds = json.loads(Path(args.pred_json).read_text())

    # image sizes
    im_wh = {im["file_name"]: (im["width"], im["height"]) for im in coco["images"]}
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}

    # GT per file
    gt_by_fname = defaultdict(list)
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        gt_by_fname[fname].append(ann)

    # Pred per file 
    pred_by_fname = {p["image_id"]: p["detections"] for p in preds}

    # Prebuild per-image GT boxes (xyxy pixels)
    gt_xyxy = {}
    for fname, anns in gt_by_fname.items():
        boxes = [coco_xywh_to_xyxy(a["bbox"]) for a in anns]
        gt_xyxy[fname] = boxes

    # Prebuild per-image predicted boxes (xyxy pixels) + scores
    pred_xyxy_scores = {}
    for fname, dets in pred_by_fname.items():
        if fname not in im_wh:
            continue
        W, H = im_wh[fname]
        items = []
        for d in dets:
            s = float(d["score"])
            box_xyxy = cxcywh_norm_to_xyxy(d["bbox"], W, H)
            items.append((s, box_xyxy))
        # sort score desc
        items.sort(key=lambda x: -x[0])
        pred_xyxy_scores[fname] = items

    thresholds = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    Ps, Rs, F1s = [], [], []

    # Evaluate each threshold
    for thr in thresholds:
        TP = FP = FN = 0
        for fname in gt_xyxy.keys():
            gtb = gt_xyxy.get(fname, [])
            preds_list = pred_xyxy_scores.get(fname, [])
            # filter preds by threshold
            pb = [b for (s, b) in preds_list if s >= thr]

            tp, fp, fn = greedy_match_counts(gtb, pb, args.iou_thr)
            TP += tp
            FP += fp
            FN += fn

        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)

    Ps = np.array(Ps)
    Rs = np.array(Rs)
    F1s = np.array(F1s)

    best_i = int(np.argmax(F1s))
    best_thr = float(thresholds[best_i])
    best_f1 = float(F1s[best_i])

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["threshold,precision,recall,f1"]
    for t, p, r, f in zip(thresholds, Ps, Rs, F1s):
        lines.append(f"{t:.6f},{p:.6f},{r:.6f},{f:.6f}")
    out_csv.write_text("\n".join(lines))
    print("Wrote:", out_csv)

    # Plot
    plt.figure()
    plt.plot(thresholds, Ps, label="Precision")
    plt.plot(thresholds, Rs, label="Recall")
    plt.plot(thresholds, F1s, label="F1")
    plt.axvline(best_thr, linestyle="--", label=f"Best thr={best_thr:.2f} (F1={best_f1:.3f})")
    plt.xlabel("Score threshold")
    plt.ylabel("Metric")
    plt.title(f"Detection VAL: Precision/Recall/F1 vs threshold @ IoU={args.iou_thr}")
    plt.legend()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote:", out_png)
    print(f"Best threshold: {best_thr:.4f} | F1: {best_f1:.6f}")


if __name__ == "__main__":
    main()
