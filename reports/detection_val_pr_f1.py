import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def iou_xywh(a, b):
    # a,b: (4,) xywh in pixels
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_gt", required=True, help="COCO val annotations json")
    ap.add_argument("--pred_json", required=True, help="Predictions json from your model")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_min", type=float, default=0.0)
    ap.add_argument("--score_max", type=float, default=1.0)
    ap.add_argument("--score_steps", type=int, default=51)
    ap.add_argument("--bbox_norm", type=int, default=1, help="1 if bbox cxcywh is normalized [0..1], else 0")
    ap.add_argument("--out_fig_dir", default="reports/figures")
    ap.add_argument("--out_tbl_dir", default="reports/tables")
    args = ap.parse_args()

    out_fig = Path(args.out_fig_dir); out_fig.mkdir(parents=True, exist_ok=True)
    out_tbl = Path(args.out_tbl_dir); out_tbl.mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(args.coco_gt).read_text())
    preds = json.loads(Path(args.pred_json).read_text())

    # map image filename -> (id, width, height)
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    fname_to_img = {}
    for im in images:
        fname_to_img[im["file_name"]] = im

    # category name -> id
    catname_to_id = {c["name"]: c["id"] for c in cats}

    # GT per image_id: list of (cat_id, bbox_xywh)
    gt = {}
    for a in anns:
        img_id = a["image_id"]
        gt.setdefault(img_id, []).append((a["category_id"], a["bbox"]))  # bbox already xywh pixels

    # Pred per image_id: list of (cat_id, score, bbox_xywh)
    pred_by_imgid = {}

    for item in preds:
        fname = item["image_id"]
        if fname not in fname_to_img:
            # skip if not in this COCO split
            continue

        im = fname_to_img[fname]
        img_id = im["id"]
        W, H = im["width"], im["height"]

        dets = []
        for d in item.get("detections", []):
            cname = d["category"]
            if cname not in catname_to_id:
                continue
            cid = catname_to_id[cname]
            score = float(d["score"])
            cx, cy, w, h = d["bbox"]

            if args.bbox_norm == 1:
                cx, cy, w, h = cx * W, cy * H, w * W, h * H

            # convert cxcywh -> xywh
            x = cx - w / 2.0
            y = cy - h / 2.0
            dets.append((cid, score, (x, y, w, h)))

        pred_by_imgid[img_id] = dets

    # Threshold sweep to compute precision/recall/F1 (micro-averaged)
    thresholds = np.linspace(args.score_min, args.score_max, args.score_steps)

    precisions = []
    recalls = []
    f1s = []

    # For matching, we do greedy per image per class
    for thr in thresholds:
        TP = 0
        FP = 0
        FN = 0

        for im in images:
            img_id = im["id"]
            gt_list = gt.get(img_id, [])
            pred_list = pred_by_imgid.get(img_id, [])

            # filter preds by threshold
            pred_list = [p for p in pred_list if p[1] >= thr]

            # group by class
            gt_by_c = {}
            for cid, bbox in gt_list:
                gt_by_c.setdefault(cid, []).append(bbox)

            pred_by_c = {}
            for cid, score, bbox in pred_list:
                pred_by_c.setdefault(cid, []).append((score, bbox))

            # greedy match per class
            for cid, gt_boxes in gt_by_c.items():
                used = [False] * len(gt_boxes)
                preds_c = sorted(pred_by_c.get(cid, []), key=lambda x: -x[0])

                for score, pb in preds_c:
                    best_iou = 0.0
                    best_j = -1
                    for j, gb in enumerate(gt_boxes):
                        if used[j]:
                            continue
                        iou = iou_xywh(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_iou >= args.iou_thr and best_j >= 0:
                        TP += 1
                        used[best_j] = True
                    else:
                        FP += 1

                # leftover GT are FN
                FN += sum(1 for u in used if not u)

            # also count preds whose class doesn't exist in GT of that image as FP
            gt_classes = set([cid for cid, _ in gt_list])
            for cid, preds_c in pred_by_c.items():
                if cid not in gt_classes:
                    FP += len(preds_c)

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])
    best_p = float(precisions[best_idx])
    best_r = float(recalls[best_idx])

    # PR curve
    plt.figure()
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (IoU={args.iou_thr:.2f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = out_fig / "detection_val_pr_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # F1 vs threshold
    plt.figure()
    plt.plot(thresholds, f1s)
    plt.xlabel("Score threshold")
    plt.ylabel("F1")
    plt.title(f"F1 vs score threshold (IoU={args.iou_thr:.2f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    f1_path = out_fig / "detection_val_f1_vs_threshold.png"
    plt.savefig(f1_path, dpi=200)
    plt.close()

    # Save summary
    summary = (
        f"iou_thr={args.iou_thr}\n"
        f"best_thr={best_thr:.4f}\n"
        f"best_f1={best_f1:.6f}\n"
        f"precision_at_best={best_p:.6f}\n"
        f"recall_at_best={best_r:.6f}\n"
        f"pred_json={args.pred_json}\n"
        f"coco_gt={args.coco_gt}\n"
        f"bbox_norm={args.bbox_norm}\n"
    )
    (out_tbl / "detection_val_pr_f1_summary.txt").write_text(summary)

    print("Wrote:", pr_path)
    print("Wrote:", f1_path)
    print("Wrote:", out_tbl / "detection_val_pr_f1_summary.txt")
    print("Best threshold:", best_thr, "| F1:", best_f1, "| P:", best_p, "| R:", best_r)

if __name__ == "__main__":
    main()
