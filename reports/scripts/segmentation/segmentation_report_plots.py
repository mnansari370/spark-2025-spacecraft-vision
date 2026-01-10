import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from reports.scripts.common.paths import repo_root, report_out_latest


def safe_load_mask(p: Path) -> np.ndarray | None:
    try:
        if p.stat().st_size == 0:
            return None
        arr = np.array(Image.open(p))
    except Exception:
        return None

    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default="inference_results/segmentation/predicted_masks",
                    help="Folder containing test_XXXXX_layer.png (relative to repo root)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, use only first N masks")
    ap.add_argument("--sample_vis", type=int, default=20, help="How many masks to sample for visualization grid")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    repo = repo_root()
    out_fig, out_tab = report_out_latest()

    mask_dir = (repo / args.mask_dir).resolve()
    pngs = sorted(mask_dir.glob("test_*_layer.png"))

    print("Mask dir:", mask_dir)
    print("Found masks:", len(pngs))

    if args.limit and args.limit > 0:
        pngs = pngs[: args.limit]

    if len(pngs) == 0:
        raise RuntimeError(
            "No masks found.\n"
            f"Expected PNGs in: {mask_dir}\n"
            "Run segmentation inference first.\n"
        )

    total_counts = np.zeros(3, dtype=np.int64)  # bg, body, panels
    per_image_ratios = []
    good_files = []

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
        raise RuntimeError("All masks were unreadable or empty. Check inference output.")

    per_image_ratios = np.array(per_image_ratios, dtype=np.float32)

    # Plot: total pixel counts
    plt.figure()
    plt.bar(["bg", "body", "panels"], total_counts)
    plt.title("Segmentation pixel counts (all masks used)")
    plt.ylabel("pixel count")
    plt.tight_layout()
    plt.savefig(out_fig / "segmentation_pixel_counts.png", dpi=200)
    plt.close()

    # Plot: body/panels ratio hist
    plt.figure()
    plt.hist(per_image_ratios[:, 0], bins=40)
    plt.title("Per-image body pixel ratio")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_fig / "segmentation_body_ratio_hist.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(per_image_ratios[:, 1], bins=40)
    plt.title("Per-image panels pixel ratio")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_fig / "segmentation_panels_ratio_hist.png", dpi=200)
    plt.close()

    # Summary txt
    total_px = int(total_counts.sum())
    body_ratio = float(total_counts[1] / max(1, total_px))
    panels_ratio = float(total_counts[2] / max(1, total_px))

    (out_tab / "segmentation_summary.txt").write_text(
        f"Masks used: {len(good_files)} / {len(pngs)}\n"
        f"Total pixels: {total_px}\n"
        f"bg pixels: {int(total_counts[0])}\n"
        f"body pixels: {int(total_counts[1])} (ratio={body_ratio:.6f})\n"
        f"panels pixels: {int(total_counts[2])} (ratio={panels_ratio:.6f})\n"
    )

    # Mask grid (quick visual)
    rng = np.random.default_rng(args.seed)
    k = min(args.sample_vis, len(good_files))
    picks = rng.choice(len(good_files), size=k, replace=False)

    cols = 5
    rows = int(np.ceil(k / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(picks, 1):
        p = good_files[int(idx)]
        m = safe_load_mask(p)
        plt.subplot(rows, cols, i)
        plt.imshow(m, vmin=0, vmax=2)
        plt.title(p.name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_fig / "segmentation_mask_grid.png", dpi=200)
    plt.close()

    print("Wrote figures to:", out_fig)
    print("Wrote tables to:", out_tab)


if __name__ == "__main__":
    main()
