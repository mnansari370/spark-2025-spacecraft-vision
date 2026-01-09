import argparse
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default="inference_results/segmentation/predicted_masks")
    ap.add_argument("--out_dir", default="reports/figures")
    ap.add_argument("--tables_dir", default="reports/tables")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample_vis", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo = find_repo_root()
    mask_dir = repo / args.mask_dir
    out_dir = repo / args.out_dir
    tables_dir = repo / args.tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(mask_dir.glob("test_*_layer.png"))
    if args.limit > 0:
        masks = masks[: args.limit]

    total_counts = np.zeros(3, dtype=np.int64)
    per_image_ratios = []

    for p in masks:
        arr = np.array(Image.open(p))
        if arr.ndim != 2:
            arr = arr[..., 0]

        for k in [0, 1, 2]:
            total_counts[k] += (arr == k).sum()

        denom = arr.size
        per_image_ratios.append([
            (arr == 0).sum() / denom,
            (arr == 1).sum() / denom,
            (arr == 2).sum() / denom,
        ])

    per_image_ratios = np.array(per_image_ratios)

    # Pixel counts
    plt.figure()
    plt.bar(["bg", "body", "panels"], total_counts)
    plt.title("Segmentation pixel counts")
    plt.savefig(out_dir / "segmentation_pixel_counts.png", dpi=200)
    plt.close()

    # Ratio histograms
    labels = ["bg", "body", "panels"]
    for i, lab in enumerate(labels):
        plt.figure()
        plt.hist(per_image_ratios[:, i], bins=40)
        plt.title(f"Per-image ratio: {lab}")
        plt.savefig(out_dir / f"segmentation_ratio_hist_{i}.png", dpi=200)
        plt.close()

    # CSV summary
    with open(tables_dir / "segmentation_ratio_summary.csv", "w") as f:
        f.write("class,mean,median\n")
        for i, lab in enumerate(labels):
            f.write(f"{lab},{per_image_ratios[:,i].mean():.6f},{np.median(per_image_ratios[:,i]):.6f}\n")

    # Optional grid
    if args.sample_vis > 0:
        if args.seed:
            random.seed(args.seed)
        chosen = random.sample(masks, min(args.sample_vis, len(masks)))

        cols = min(5, len(chosen))
        rows = (len(chosen) + cols - 1) // cols
        plt.figure(figsize=(cols * 3, rows * 3))

        for i, p in enumerate(chosen, 1):
            arr = np.array(Image.open(p))
            if arr.ndim != 2:
                arr = arr[..., 0]
            plt.subplot(rows, cols, i)
            plt.imshow(arr)
            plt.axis("off")
            plt.title(p.name, fontsize=8)

        plt.savefig(out_dir / "segmentation_sample_masks_grid.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
