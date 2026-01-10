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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_gt", default="data/annotations/spark_val.json")
    ap.add_argument("--pred_json", default="inference_results/detection/val_predictions.json")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_thr", type=float, default=0.94, help="use best threshold from PR/F1")
    ap.add_argument("--out_table", default="reports/tables/detection_val_per_class_metrics.csv")
    ap.add_argument("--out_bar", default="reports/figures/detection_val_per_class_f1.png")
    ap.add_argument("--out_confmat", default="reports/figures/detection_val_confusion_like.png")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco_gt).read_text())
    preds = json.loads(Path(args.pred_json).read_text())

    # categories
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    name2id = {v: k for k, v in id2name.items()}
    class_names = [id2name[i] for i in sorted(id2name.keys())]
    C = len(class_names)

    # image sizes and image_id mapping
    im_wh = {}
    imgid_by_fname = {}
    for im in coco["images"]:
        im_wh[im["file_name"]] = (im["width"], im["height"])
        imgid_by_fname[im["file_name"]] = im["id"]

    # GT annotations per file_name
    gt_by_fname = defaultdict(list)
    for ann in coco["annotations"]:
        # need file_name; map ann.image_id -> file_name
        # build reverse map once
        pass
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        gt_by_fname[fname].append(ann)

    # Preds per file_name (your pred uses image_id as file_name)
    pred_by_fname = {p["image_id"]: p["detections"] for p in preds}

    # stats
    tp = np.zeros(C, dtype=int)
    fp = np.zeros(C, dtype=int)
    fn = np.zeros(C, dtype=int)
    iou_sum = np.zeros(C, dtype=float)
    iou_cnt = np.zeros(C, dtype=int)

    conf = np.zeros((C, C), dtype=int)  # GT row, Pred col (matched only)

    # match per image with greedy matching by IoU (per GT)
    for fname, gt_anns in gt_by_fname.items():
        W, H = im_wh[fname]
        gt_boxes = []
        gt_cls = []
        for a in gt_anns:
            gt_boxes.append(coco_xywh_to_xyxy(a["bbox"]))
            gt_cls.append(int(a["category_id"]))

        dets = pred_by_fname.get(fname, [])
        # filter by score_thr and convert bbox to xyxy
        pred_boxes = []
        pred_cls = []
        pred_score = []
        for d in dets:
            if float(d["score"]) < args.score_thr:
                continue
            cname = d["category"]
            if cname not in name2id:
                continue
            pred_cls.append(name2id[cname])
            pred_score.append(float(d["score"]))
            pred_boxes.append(cxcywh_norm_to_xyxy(d["bbox"], W, H))

        # if no GT and preds exist -> all preds are FP (but dataset may have empty images)
        if len(gt_boxes) == 0:
            for pc in pred_cls:
                fp[pc] += 1
            continue

        # greedy match: for each pred, find best GT; enforce 1-1 matching
        matched_gt = set()
        matched_pred = set()

        # sort preds by score desc (standard)
        order = np.argsort(-np.array(pred_score)) if len(pred_score) else []
        for pi in order:
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                iou = iou_xyxy(pred_boxes[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou >= args.iou_thr:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

                gcls = gt_cls[best_gi]
                pcls = pred_cls[pi]
                tp[pcls] += 1
                iou_sum[pcls] += best_iou
                iou_cnt[pcls] += 1
                conf[gcls, pcls] += 1
            else:
                # unmatched pred -> FP
                fp[pred_cls[pi]] += 1

        # unmatched GT -> FN (counted by GT class)
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn[gt_cls[gi]] += 1

    # per-class metrics
    prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(prec), where=(prec + rec) > 0)
    miou = np.divide(iou_sum, iou_cnt, out=np.zeros_like(iou_sum), where=iou_cnt > 0)

    # write table
    out_table = Path(args.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    lines = ["class,tp,fp,fn,precision,recall,f1,mean_iou_tp"]
    for cid, cname in enumerate(class_names):
        lines.append(
            f"{cname},{tp[cid]},{fp[cid]},{fn[cid]},"
            f"{prec[cid]:.6f},{rec[cid]:.6f},{f1[cid]:.6f},{miou[cid]:.6f}"
        )
    out_table.write_text("\n".join(lines))
    print("Wrote:", out_table)

    # bar chart (F1 per class)
    plt.figure(figsize=(10, 4))
    x = np.arange(C)
    plt.bar(x, f1)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("F1")
    plt.title(f"Detection VAL: per-class F1 @ IoU={args.iou_thr} thr={args.score_thr}")
    out_bar = Path(args.out_bar)
    out_bar.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_bar, dpi=200, bbox_inches="tight")
    print("Wrote:", out_bar)

    # confusion-like matrix (matched only) plotted as image
    plt.figure(figsize=(8, 7))
    plt.imshow(conf, aspect="auto")
    plt.xticks(np.arange(C), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(C), class_names)
    plt.xlabel("Predicted class (matched)")
    plt.ylabel("GT class (matched)")
    plt.title("Detection VAL: confusion-like matrix (matched TPs only)")
    plt.colorbar()
    out_cm = Path(args.out_confmat)
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_cm, dpi=200, bbox_inches="tight")
    print("Wrote:", out_cm)


if __name__ == "__main__":
    main()
