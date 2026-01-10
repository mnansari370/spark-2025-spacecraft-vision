#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# repo root: scripts/segmentation/ -> scripts/ -> repo
REPO_ROOT = Path(__file__).resolve().parents[2]

# colors: body=red, panels=blue
RED = (255, 0, 0)
BLU = (0, 0, 255)


def load_mask_from_npz(npz_path: Path) -> np.ndarray:
    """
    Supports two formats:
      1) keys: ['layer'] -> uint8 label image (H,W) values {0,1,2}
      2) keys: ['data']  -> bool array (H,W,3):
            data[...,0] = body
            data[...,2] = panels
    Returns: (H,W) uint8 label mask.
    """
    z = np.load(npz_path)

    if "layer" in z:
        lab = z["layer"].astype(np.uint8)
        if lab.ndim != 2:
            raise ValueError(f"Unexpected 'layer' shape in {npz_path.name}: {lab.shape}")
        return lab

    if "data" in z:
        data = z["data"].astype(bool)
        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError(f"Unexpected 'data' shape in {npz_path.name}: {data.shape}")

        body = data[..., 0]
        panels = data[..., 2]

        lab = np.zeros(body.shape, dtype=np.uint8)
        lab[body] = 1
        lab[panels] = 2
        return lab

    raise ValueError(f"Unknown keys in {npz_path.name}: {list(z.keys())}")


def label_to_color_image(lab: np.ndarray) -> np.ndarray:
    """Convert label mask (0/1/2) -> RGB visualization."""
    h, w = lab.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    body = (lab == 1)
    panels = (lab == 2)

    out[body] = RED
    out[panels] = BLU
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz_dir",
        default="inference_results/segmentation/npz_tmp",
        help="Folder containing test_XXXXX_layer.npz files",
    )
    ap.add_argument(
        "--out_dir",
        default="inference_results/segmentation/visuals",
        help="Where to write visualization PNGs",
    )
    ap.add_argument("--n", type=int, default=40, help="Export top-N masks by foreground pixels")
    args = ap.parse_args()

    npz_dir = (REPO_ROOT / args.npz_dir).expanduser().resolve()
    out_dir = (REPO_ROOT / args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_dir.exists():
        raise FileNotFoundError(f"npz_dir not found: {npz_dir}")

    npz_files = sorted(npz_dir.glob("test_*_layer.npz"))
    print(f"NPZ files found: {len(npz_files)} in {npz_dir}")

    if not npz_files:
        print("Nothing to visualize (npz_dir is empty).")
        return

    # score by foreground pixels (body + panels)
    scored = []
    for f in npz_files:
        try:
            lab = load_mask_from_npz(f)
            fg = int((lab == 1).sum() + (lab == 2).sum())
            scored.append((fg, f))
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}")

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: min(args.n, len(scored))]

    print(f"Exporting {len(chosen)} masks -> {out_dir}")

    saved = 0
    for fg, f in chosen:
        try:
            lab = load_mask_from_npz(f)
            rgb = label_to_color_image(lab)
            img = Image.fromarray(rgb, mode="RGB")

            out_name = f"{f.stem.replace('_layer', '')}_seg.png"  # test_00000_seg.png
            img.save(out_dir / out_name)
            saved += 1
        except Exception as e:
            print(f"[WARN] failed {f.name}: {e}")

    print(f"[OK] Saved {saved}/{len(chosen)} visuals -> {out_dir}")


if __name__ == "__main__":
    main()
