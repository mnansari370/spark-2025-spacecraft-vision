# reports/scripts/segmentation/segmentation_val_metrics.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from reports.scripts.common.paths import repo_root, report_out_latest


def iter_mask_files(root: Path):
    """Recursively yield mask files (png/jpg/jpeg)."""
    exts = {".png", ".jpg", ".jpeg"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def collect_layer_masks(root: Path):
    """Collect only *_layer.* masks recursively."""
    files = [p for p in iter_mask_files(root) if "_layer" in p.name]
    return sorted(files)


def load_mask(path: Path) -> np.ndarray:
    """
    Load a mask image as int64 2D array.
    If image is RGB, take channel 0.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def fast_confusion(gt: np.ndarray, pr: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Fast confusion matrix using bincount.
    conf[i,j] = count of pixels where GT=i and Pred=j
    """
    gt = gt.reshape(-1)
    pr = pr.reshape(-1)

    valid = (gt >= 0) & (gt < num_classes) & (pr >= 0) & (pr < num_classes)
    gt = gt[valid]
    pr = pr[valid]

    idx = gt * num_classes + pr
    conf = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf.astype(np.int64)


def compute_ious(conf: np.ndarray):
    """IoU per class + mIoU from confusion matrix."""
    num_classes = conf.shape[0]
    ious = []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        iou = (tp / denom) if denom > 0 else 0.0
        ious.append(float(iou))
    miou = float(np.mean(ious)) if num_classes > 0 else 0.0
    return ious, miou


def plot_confusion(conf: np.ndarray, class_names, out_path: Path):
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    im = ax.imshow(conf.astype(np.float64), interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_title("Segmentation Confusion Matrix (pixels)\nrows=GT, cols=Pred")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Write values (matrix is small)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, str(int(conf[i, j])), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_iou_bar(ious, class_names, out_path: Path):
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.bar(range(len(ious)), ious)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title("IoU per class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="GT masks root (recursive). GT can be .jpg/.png")
    ap.add_argument("--pred_dir", required=True, help="Pred masks root (recursive). Pred typically .png")
    ap.add_argument("--limit", type=int, default=0, help="0 = all, else only first N matched pairs")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument(
        "--class_names",
        nargs="*",
        default=None,
        help="Optional class names list length=num_classes (e.g., bg body panels)",
    )
    args = ap.parse_args()

    repo = repo_root()
    out_fig, out_tab = report_out_latest()

    gt_root = Path(args.gt_dir)
    pr_root = Path(args.pred_dir)

    # Allow relative paths (relative to repo root)
    if not gt_root.is_absolute():
        gt_root = (repo / gt_root).resolve()
    if not pr_root.is_absolute():
        pr_root = (repo / pr_root).resolve()

    if not gt_root.exists():
        raise RuntimeError(f"GT dir does not exist: {gt_root}")
    if not pr_root.exists():
        raise RuntimeError(f"Pred dir does not exist: {pr_root}")

    gt_files = collect_layer_masks(gt_root)
    pr_files = collect_layer_masks(pr_root)

    if len(gt_files) == 0:
        raise RuntimeError(f"No GT *_layer masks found in: {gt_root} (recursive search)")
    if len(pr_files) == 0:
        raise RuntimeError(f"No predicted *_layer masks found in: {pr_root} (recursive search)")

    # Match by STEM so .jpg matches .png (image_00007_layer)
    gt_by_stem = {p.stem: p for p in gt_files}
    pr_by_stem = {p.stem: p for p in pr_files}

    common = sorted(set(gt_by_stem.keys()) & set(pr_by_stem.keys()))
    if len(common) == 0:
        raise RuntimeError(
            "No overlap between GT and predicted mask stems.\n"
            f"GT example: {gt_files[0].name} (stem={gt_files[0].stem})\n"
            f"Pred example: {pr_files[0].name} (stem={pr_files[0].stem})\n"
            "Expected same stem like image_00007_layer"
        )

    pairs = [(gt_by_stem[k], pr_by_stem[k]) for k in common]
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    # Class names
    if args.class_names is not None and len(args.class_names) == args.num_classes:
        class_names = args.class_names
    else:
        # default for this challenge
        class_names = ["bg", "body", "panels"] if args.num_classes == 3 else [f"class_{i}" for i in range(args.num_classes)]

    total_conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    used = 0
    skipped_shape = 0

    for gt_p, pr_p in pairs:
        gt = load_mask(gt_p)
        pr = load_mask(pr_p)

        if gt.shape != pr.shape:
            skipped_shape += 1
            continue

        total_conf += fast_confusion(gt, pr, args.num_classes)
        used += 1

    ious, miou = compute_ious(total_conf)

    # Write TXT
    txt_lines = []
    txt_lines.append("Segmentation VAL metrics")
    txt_lines.append(f"GT dir:   {gt_root}")
    txt_lines.append(f"Pred dir: {pr_root}")
    txt_lines.append("")
    txt_lines.append(f"Matched pairs total: {len(pairs)}")
    txt_lines.append(f"Pairs used (same shape): {used}")
    txt_lines.append(f"Pairs skipped (shape mismatch): {skipped_shape}")
    txt_lines.append("")
    txt_lines.append("Confusion matrix (rows=GT, cols=Pred) [pixel counts]:")
    txt_lines.append(str(total_conf))
    txt_lines.append("")
    txt_lines.append("IoU per class:")
    for name, val in zip(class_names, ious):
        txt_lines.append(f"{name}: {val:.6f}")
    txt_lines.append(f"mIoU: {miou:.6f}")

    (out_tab / "segmentation_val_metrics.txt").write_text("\n".join(txt_lines))
    print("Wrote:", out_tab / "segmentation_val_metrics.txt")

    # Write IoU CSV
    csv_lines = ["class,iou"]
    for name, val in zip(class_names, ious):
        csv_lines.append(f"{name},{val:.6f}")
    csv_lines.append(f"mIoU,{miou:.6f}")
    (out_tab / "segmentation_val_iou.csv").write_text("\n".join(csv_lines))
    print("Wrote:", out_tab / "segmentation_val_iou.csv")

    # Plots
    plot_confusion(total_conf, class_names, out_fig / "segmentation_val_confusion_matrix.png")
    print("Wrote:", out_fig / "segmentation_val_confusion_matrix.png")

    plot_iou_bar(ious, class_names, out_fig / "segmentation_val_iou_bar.png")
    print("Wrote:", out_fig / "segmentation_val_iou_bar.png")

    print("\nOutputs:")
    print("Figures:", out_fig)
    print("Tables :", out_tab)


if __name__ == "__main__":
    main()
