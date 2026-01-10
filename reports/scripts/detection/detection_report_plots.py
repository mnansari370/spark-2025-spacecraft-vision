import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from reports.scripts.common.paths import repo_root, report_out_latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_json",
        default="inference_results/detection/detection_test_predictions.json",
        help="Path to detection predictions JSON (relative to repo root)",
    )
    ap.add_argument("--score_thr", type=float, default=0.0, help="Filter detections below this score")
    args = ap.parse_args()

    repo = repo_root()
    out_fig, out_tab = report_out_latest()

    pred_json = (repo / args.pred_json).resolve()
    if not pred_json.exists():
        raise FileNotFoundError(f"Missing predictions JSON: {pred_json}")

    data = json.loads(pred_json.read_text())

    n_per_img = []
    all_scores = []
    class_counts = Counter()
    rows_top = []

    for item in data:
        img_id = item.get("image_id")
        dets = item.get("detections", [])
        dets = [d for d in dets if float(d.get("score", 0.0)) >= args.score_thr]

        n_per_img.append(len(dets))

        if dets:
            top = max(dets, key=lambda d: float(d.get("score", 0.0)))
            rows_top.append((img_id, top.get("category", ""), float(top.get("score", 0.0))))

        for d in dets:
            all_scores.append(float(d.get("score", 0.0)))
            class_counts[d.get("category", "UNKNOWN")] += 1

    n_per_img = np.array(n_per_img)
    all_scores = np.array(all_scores)

    # Detections per image
    plt.figure()
    plt.hist(n_per_img, bins=30)
    plt.title(f"Detections per image (thr={args.score_thr})")
    plt.xlabel("# detections")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_fig / "detection_dets_per_image_hist.png", dpi=200)
    plt.close()

    # Score distribution
    if len(all_scores) > 0:
        plt.figure()
        plt.hist(all_scores, bins=40)
        plt.title(f"Detection score distribution (thr={args.score_thr})")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_fig / "detection_score_hist.png", dpi=200)
        plt.close()

    # Class counts
    if class_counts:
        plt.figure()
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Predicted class counts (thr={args.score_thr})")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_fig / "detection_class_counts.png", dpi=200)
        plt.close()

    # CSV table: top-1 per image
    rows_top.sort(key=lambda x: x[2], reverse=True)
    csv_path = out_tab / "detection_top1_per_image.csv"
    with open(csv_path, "w") as f:
        f.write("image_id,top_category,top_score\n")
        for r in rows_top:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f}\n")

    # Summary
    (out_tab / "detection_summary.txt").write_text(
        f"images: {len(n_per_img)}\n"
        f"mean_dets_per_image: {n_per_img.mean():.3f}\n"
        f"median_dets_per_image: {np.median(n_per_img):.3f}\n"
        f"total_detections: {len(all_scores)}\n"
    )

    print("Wrote figures to:", out_fig)
    print("Wrote tables to:", out_tab)


if __name__ == "__main__":
    main()
