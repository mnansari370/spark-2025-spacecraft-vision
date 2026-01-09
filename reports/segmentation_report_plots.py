import argparse
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default="inference_results/segmentation/predicted_masks")
    ap.add_argument("--out_dir", default="reports/figures")
    ap.add_argument("--tables_dir", default="reports/tables")
    ap.add_argument("--limit", type=int, default=0, help="Use only first N masks (0 = all)")
    ap.add_argument("--sample_vis", type=int, default=0, help="Save a grid of N masks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo = repo_root()
    mask_dir = (repo / args.mask_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()
    tables_dir = (repo / args.tables_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir does not exist: {mask_dir}")

    masks = sorted(mask_dir.glob("test_*_layer.png"))
    print("Mask dir:", mask_dir)
    print("Found masks:", len(masks))

    if args.limit and args.limit > 0:
        masks = masks[: args.limit]
        print("Using first masks (limit):", len(masks))

    if len(masks) == 0:
        raise RuntimeError(
            "No masks found. "
            "Make sure you ran segmentation inference and PNGs exist in mask_dir."
        )

    total_counts = np.zeros(3, dtype=np.int64)
    per_image_ratios = []

    bad = 0
    for p in masks:
        try:
            arr = np.array(Image.open(p))
        except Exception as e:
            bad += 1
            print("SKIP bad png:", p.name, "|", e)
            continue

        if arr.ndim != 2:
            arr = arr[..., 0]

        # counts per class
        denom = int(arr.size)
        if denom == 0:
            bad += 1
            print("SKIP empty mask:", p.name)
            continue

        c0 = int((arr == 0).sum())
        c1 = int((arr == 1).sum())
        c2 = int((arr == 2).sum())

        total_counts[0] += c0
        total_counts[1] += c1
        total_counts[2] += c2

        per_image_ratios.append([c0 / denom, c1 / denom, c2 / denom])

    if len(per_image_ratios) == 0:
        raise RuntimeError("All masks were skipped as bad/empty. Nothing to plot.")

    per_image_ratios = np.asarray(per_image_ratios, dtype=np.float64)
    print("Used masks:", per_image_ratios.shape[0], "| skipped:", bad)
    print("per_image_ratios shape:", per_image_ratios.shape)

    # Pixel counts plot
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
        plt.xlabel("ratio")
        plt.ylabel("count")
        plt.savefig(out_dir / f"segmentation_ratio_hist_{i}.png", dpi=200)
        plt.close()

    # CSV summary
    csv_path = tables_dir / "segmentation_ratio_summary.csv"
    with open(csv_path, "w") as f:
        f.write("class,mean,median\n")
        for i, lab in enumerate(labels):
            f.write(f"{lab},{per_image_ratios[:, i].mean():.6f},{np.median(per_image_ratios[:, i]):.6f}\n")
    print("Wrote:", csv_path)

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

        out_grid = out_dir / "segmentation_sample_masks_grid.png"
        plt.savefig(out_grid, dpi=200)
        plt.close()
        print("Wrote:", out_grid)


if __name__ == "__main__":
    main()
