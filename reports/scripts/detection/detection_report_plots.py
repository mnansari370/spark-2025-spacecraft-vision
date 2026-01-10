import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the spark_project root.
    We detect it by the presence of key folders/files.
    """
    start = start.resolve()
    markers = ["README.md", "run", "data", "inference_results", "reports"]
    for d in [start] + list(start.parents):
        ok = True
        for m in markers:
            if not (d / m).exists():
                ok = False
                break
        if ok:
            return d
    # fallback: assume file lives under <repo>/reports/scripts/...
    return start.parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_json",
        default="inference_results/detection/detection_test_predictions.json",
        help="Path to detection predictions JSON (relative to repo root)",
    )
    ap.add_argument("--out_dir", default="reports/figures", help="Where to save plots (relative to repo root)")
    ap.add_argument("--tables_dir", default="reports/tables", help="Where to save tables (relative to repo root)")
    ap.add_argument("--score_thr", type=float, default=0.0, help="Filter detections below this score")
    args = ap.parse_args()

    repo = find_repo_root(Path(__file__).resolve())
    pred_json = (repo / args.pred_json).resolve()
    out_dir = (repo / args.out_dir).resolve()
    tables_dir = (repo / args.tables_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Repo root:", repo)
    print("Pred JSON:", pred_json)

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

    n_per_img = np.array(n_per_img, dtype=np.int64)
    all_scores = np.array(all_scores, dtype=np.float32)

    # Histogram: detections per image
    plt.figure()
    plt.hist(n_per_img, bins=30)
    plt.title(f"Detections per image (thr={args.score_thr})")
    plt.xlabel("# detections")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "detection_dets_per_image_hist.png", dpi=200)
    plt.close()

    # Histogram: score distribution
    if len(all_scores) > 0:
        plt.figure()
        plt.hist(all_scores, bins=40)
        plt.title(f"Detection score distribution (thr={args.score_thr})")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "detection_score_hist.png", dpi=200)
        plt.close()

    # Bar: class counts
    if class_counts:
        plt.figure()
        plt.bar(list(class_counts.keys()), list(class_counts.values()))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Predicted class counts (thr={args.score_thr})")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "detection_class_counts.png", dpi=200)
        plt.close()

    # CSV: top1 per image
    rows_top.sort(key=lambda x: x[2], reverse=True)
    csv_path = tables_dir / "detection_top1_per_image.csv"
    with open(csv_path, "w") as f:
        f.write("image_id,top_category,top_score\n")
        for r in rows_top:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f}\n")

    # TXT summary
    with open(tables_dir / "detection_summary.txt", "w") as f:
        f.write(f"images: {len(n_per_img)}\n")
        f.write(f"mean_dets_per_image: {float(n_per_img.mean()):.3f}\n")
        f.write(f"median_dets_per_image: {float(np.median(n_per_img)):.3f}\n")
        f.write(f"total_detections: {len(all_scores)}\n")

    print("Wrote figures to:", out_dir)
    print("Wrote tables  to:", tables_dir)


if __name__ == "__main__":
    main()
