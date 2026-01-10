# reports/scripts/segmentation/segmentation_report_plots.py
from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def safe_load_mask(p: Path) -> np.ndarray | None:
    """Load a single-channel label mask as uint8 (H,W). Return None if unreadable/bad."""
    try:
        if p.stat().st_size == 0:
            return None
        arr = np.array(Image.open(p))
    except Exception:
        return None

    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr.astype(np.uint8)


def collect_masks(mask_root: Path, pattern: str, recursive: bool) -> list[Path]:
    """Collect mask files from a directory."""
    if recursive:
        return sorted(mask_root.rglob(pattern))
    return sorted(mask_root.glob(pattern))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mask_dir",
        default="inference_results/segmentation/predicted_masks",
        help="Root folder containing *_layer.png masks (supports nested folders)",
    )
    ap.add_argument(
        "--pattern",
        default="*_layer.png",
        help="Mask filename pattern (default works for both test_* and image_*).",
    )
    ap.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursive search (default: recursive ON).",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, use only first N masks")
    ap.add_argument("--sample_vis", type=int, default=25, help="How many masks to sample for a grid")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--out_fig_dir", default="reports/figures")
    ap.add_argument("--out_table_dir", default="reports/tables")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()  # run from repo => safest on HPC
    mask_dir = (repo_root / args.mask_dir).resolve()
    out_fig = (repo_root / args.out_fig_dir).resolve()
    out_tab = (repo_root / args.out_table_dir).resolve()

    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    print("Repo root:", repo_root)
    print("Mask dir:", mask_dir)

    pngs = collect_masks(mask_dir, args.pattern, recursive=(not args.no_recursive))
    print("Found masks:", len(pngs))

    if args.limit and args.limit > 0:
        pngs = pngs[: args.limit]
        print("Using masks (limit):", len(pngs))

    if len(pngs) == 0:
        raise RuntimeError(
            "No masks found.\n"
            f"Looked in: {mask_dir}\n"
            f"pattern: {args.pattern} | recursive: {not args.no_recursive}\n"
            "Tip (VAL): use --mask_dir inference_results/segmentation/val_predicted_masks\n"
        )

    # Accumulators
    total_counts = np.zeros(3, dtype=np.int64)  # bg, body, panels
    per_image_ratios = []
    good_files: list[Path] = []

    for p in pngs:
        m = safe_load_mask(p)
        if m is None:
            continue

        c0 = int((m == 0).sum())
        c1 = int((m == 1).sum())
        c2 = int((m == 2).sum())
        total_counts += np.array([c0, c1, c2], dtype=np.int64)

        denom = max(1, (c0 + c1 + c2))
        per_image_ratios.append([c1 / denom, c2 / denom])
        good_files.append(p)

    if len(good_files) == 0:
        raise RuntimeError("All masks were unreadable/empty. Check your inference output.")

    per_image_ratios = np.array(per_image_ratios, dtype=np.float32)

    # ---- Plot 1: total pixel counts (bar)
    plt.figure()
    plt.bar(["bg", "body", "panels"], total_counts)
    plt.title("Segmentation pixel counts (all masks used)")
    plt.ylabel("pixel count")
    plt.tight_layout()
    out1 = out_fig / "segmentation_pixel_counts.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print("Wrote:", out1)

    # ---- Plot 2: per-image ratios histograms
    plt.figure()
    plt.hist(per_image_ratios[:, 0], bins=40)
    plt.title("Per-image body pixel ratio")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.tight_layout()
    out2 = out_fig / "segmentation_body_ratio_hist.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    print("Wrote:", out2)

    plt.figure()
    plt.hist(per_image_ratios[:, 1], bins=40)
    plt.title("Per-image panels pixel ratio")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.tight_layout()
    out3 = out_fig / "segmentation_panels_ratio_hist.png"
    plt.savefig(out3, dpi=200)
    plt.close()
    print("Wrote:", out3)

    # ---- Table: summary txt
    total_px = int(total_counts.sum())
    body_ratio = float(total_counts[1] / max(1, total_px))
    panels_ratio = float(total_counts[2] / max(1, total_px))

    summary_txt = (
        f"Masks found: {len(pngs)}\n"
        f"Masks used:  {len(good_files)}\n"
        f"Total pixels: {total_px}\n"
        f"bg pixels: {int(total_counts[0])}\n"
        f"body pixels: {int(total_counts[1])} (ratio={body_ratio:.6f})\n"
        f"panels pixels: {int(total_counts[2])} (ratio={panels_ratio:.6f})\n"
    )
    (out_tab / "segmentation_summary.txt").write_text(summary_txt)
    print("Wrote:", out_tab / "segmentation_summary.txt")

    # ---- Grid of sample masks (raw labels)
    rng = np.random.default_rng(args.seed)
    k = min(args.sample_vis, len(good_files))
    picks = rng.choice(len(good_files), size=k, replace=False)

    cols = 5
    rows = int(math.ceil(k / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(picks, 1):
        p = good_files[int(idx)]
        m = safe_load_mask(p)
        plt.subplot(rows, cols, i)
        plt.imshow(m, vmin=0, vmax=2)
        plt.title(p.name, fontsize=7)
        plt.axis("off")

    plt.tight_layout()
    out_grid = out_fig / "segmentation_mask_grid.png"
    plt.savefig(out_grid, dpi=200)
    plt.close()
    print("Wrote:", out_grid)


if __name__ == "__main__":
    main()
