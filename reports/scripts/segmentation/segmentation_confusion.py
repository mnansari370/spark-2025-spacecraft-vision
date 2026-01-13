#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

LABELS = ["bg", "body", "panels"]
K = 3
IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_images(d: Path) -> List[Path]:
    out: List[Path] = []
    if not d.exists():
        return out
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def normalize_key(p: Path) -> str:
    """
    Make GT/pred filenames match even if suffix differs.
    Examples:
      image_00007_mask.jpg  -> 00007
      image_00007_layer.png -> 00007
      test_00007_layer.png  -> 00007
    """
    s = p.stem.lower()

    # strip common suffixes
    for suf in ["_layer", "_mask", "_pred", "_prediction", "_gt"]:
        if s.endswith(suf):
            s = s[: -len(suf)]

    # take last numeric group if present
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    if digits:
        return digits[-1].lstrip("0") or "0"
    return s


def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def compute_confusion(gt_dir: Path, pr_dir: Path, strict_shapes: bool) -> Tuple[np.ndarray, int, int]:
    gt_files = {normalize_key(x): x for x in list_images(gt_dir)}
    pr_files = {normalize_key(x): x for x in list_images(pr_dir)}

    keys = sorted(set(gt_files) & set(pr_files))
    if not keys:
        return np.zeros((K, K), dtype=np.int64), 0, 0

    conf = np.zeros((K, K), dtype=np.int64)
    used = 0

    for k in keys:
        g = read_mask(gt_files[k])
        p = read_mask(pr_files[k])

        if g.shape != p.shape:
            if strict_shapes:
                continue
            else:
                continue

        g = np.clip(g.astype(np.int64), 0, K - 1)
        p = np.clip(p.astype(np.int64), 0, K - 1)

        idx = g.reshape(-1) * K + p.reshape(-1)
        conf += np.bincount(idx, minlength=K * K).reshape(K, K)
        used += 1

    return conf, used, len(keys)


def plot_heat(mat: np.ndarray, title: str, out_path: Path, normalize: bool = False, log10: bool = False, subset=None):
    m = mat.astype(np.float64).copy()
    labs = LABELS

    if subset is not None:
        m = m[np.ix_(subset, subset)]
        labs = [LABELS[i] for i in subset]

    if normalize:
        row = m.sum(axis=1, keepdims=True)
        m = np.divide(m, row, out=np.zeros_like(m), where=row != 0)

    if log10:
        m = np.log10(m + 1.0)

    plt.figure(figsize=(7.2, 6.0), dpi=220)
    im = plt.imshow(m, aspect="equal")
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Ground Truth", fontsize=12)
    plt.xticks(range(len(labs)), labs, rotation=25, ha="right", fontsize=11)
    plt.yticks(range(len(labs)), labs, fontsize=11)

    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if normalize:
                txt = f"{m[r, c] * 100:.1f}%"
            else:
                base = mat[np.ix_(subset, subset)][r, c] if subset is not None else mat[r, c]
                txt = f"{int(base)}"
            plt.text(c, r, txt, ha="center", va="center", fontsize=9)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", default="data/spark-2024-train-val/mask", help="GT root folder")
    ap.add_argument("--pred_root", default="inference_results/segmentation/val_predicted_masks", help="Pred root folder")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out_dir", default="reports/latest")
    ap.add_argument("--strict_shapes", action="store_true")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    pr_root = Path(args.pred_root)
    out_root = Path(args.out_dir)
    out_tab = out_root / "tables"
    out_fig = out_root / "figures"

    missions = sorted([d.name for d in gt_root.iterdir() if d.is_dir()])
    if not missions:
        raise SystemExit(f"No missions found under: {gt_root}")

    total = np.zeros((K, K), dtype=np.int64)
    used_summary = []

    for m in missions:
        gt_dir = gt_root / m / args.split
        pr_dir = pr_root / m / args.split

        if not gt_dir.exists() or not pr_dir.exists():
            continue

        conf_m, used, inter = compute_confusion(gt_dir, pr_dir, args.strict_shapes)
        if conf_m.sum() == 0:
            continue

        total += conf_m
        used_summary.append((m, used, int(conf_m.sum()), inter))

    if total.sum() == 0:
        raise SystemExit("Total confusion sum is 0. Check predicted val masks exist for all missions.")

    out_tab.mkdir(parents=True, exist_ok=True)
    np.save(out_tab / "segmentation_val_confusion_pixels.npy", total)

    (out_tab / "segmentation_val_confusion_used_files.txt").write_text(
        "\n".join([f"{m}\tused={u}\tpixels={px}\tcommon_keys={inter}" for (m, u, px, inter) in used_summary]) + "\n"
    )

    plot_heat(total, "Segmentation VAL: Confusion (log10 pixels)",
              out_fig / "segmentation_val_confusion_log10.png", log10=True)
    plot_heat(total, "Segmentation VAL: Confusion (row-normalized)",
              out_fig / "segmentation_val_confusion_row_norm.png", normalize=True)
    plot_heat(total, "Segmentation VAL: Foreground confusion (row-norm)",
              out_fig / "segmentation_val_confusion_foreground_row_norm.png", normalize=True, subset=[1, 2])

    print("OK: wrote summed confusion across missions to:", out_root)
    print("Total pixels:", int(total.sum()))


if __name__ == "__main__":
    main()
